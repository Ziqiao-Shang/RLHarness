"""Offline tests for current seed creation; scripted replies are not results."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rlharness.common.config import ROOT
from rlharness.common.data_load import load_planning
from rlharness.common.prompt_build import _weights_tuple
from rlharness.data_process.sft_audit import _load_vertex_rows, compute_route_metrics
from rlharness.data_process.skillopt_split import prepare as prepare_skillopt_split
from rlharness.initialization.execution import APIBackend, bundle_for, solve, verify
from rlharness.initialization.seed import parse_args, run
from rlharness.initialization.state import assemble, extract_frame, read_json


def fixture_bank() -> dict:
    return {
        "skills": [
            {
                "skill_id": index + 1,
                "category": ("image", "table", "fusion")[index // 5],
                "title": f"Fixture skill {index + 1}",
                "when_to_apply": "During route construction",
                "procedure": "Trace the visible route",
                "check": "Check its endpoints",
            }
            for index in range(15)
        ]
    }


class ScriptedBackend:
    def __init__(self, domain: str = "travelmap"):
        self.rows = {row["sample_id"]: row for row in load_planning(domain, "train")}
        self.calls: list[tuple] = []
        self.fail_transport = False
        self.usage = "Use [Relevant Strategy Selection] and mark execution with [Using Skill N]."

    def response(self, row: dict) -> dict:
        table = _load_vertex_rows(Path(bundle_for(row, "{question}")["table_path"]))
        metrics = compute_route_metrics(
            row["gt_route"], table, list(_weights_tuple(row)), domain=row["domain"]
        )
        values = {"T": metrics.t, "P": metrics.p, "C": metrics.c,
                  "R": metrics.r, "Score": metrics.score}
        output = (
            "<reasoning>[Task and Map Analysis]\nInspect the map.\n"
            "[Relevant Strategy Selection]\n- Skill 1: trace\n- Skill 2: bind\n- Skill 3: score\n"
            "[Candidate Routes]\n[Using Skill 1] Trace the route.\n"
            "[Using Skill 2] Bind the table.\n[Score Calculation]\n"
            "[Using Skill 3] Compute "
            + ", ".join(f"{key}={value:.8f}" for key, value in values.items())
            + "\n[Decision and Verification]\nCheck endpoints.</reasoning>\n"
            + f"<response>{row['gt_route']}</response>"
        )
        return {"output": output, "metrics": values}

    def request(self, path, *, model, instruction, payload, images):
        if self.fail_transport:
            raise RuntimeError("Simulated transport outage")
        self.calls.append((str(path), model, copy.deepcopy(payload), len(images)))
        if "student_prompt" in payload:
            self._check_rule_source(instruction, payload, "student_prompt")
            if "gt_route" in payload:
                raise AssertionError("Reference route leaked into execution request")
            return self.response(self.rows[payload["sample_id"]])
        if "skill_bank" in payload:
            self._check_rule_source(instruction, payload, "fixed_frame")
            return {"how_to_use": self.usage}
        self._check_rule_source(instruction, payload, "fixed_frame")
        return fixture_bank()

    @staticmethod
    def _check_rule_source(instruction, payload, field):
        if field not in instruction or "# Authoritative Scoring Rules" not in payload[field]:
            raise AssertionError("API call lost its authoritative task rules")


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.reference = ROOT / "data" / "reference" / "travelmap" / "splits"
        self.split = self.folder / "skillopt_train480"
        prepare_skillopt_split("travelmap", self.reference, self.split)
        train_ids = read_json(self.split / "train_sample_ids.json")
        self.args = parse_args([
            "--domain", "travelmap",
            "--train-ids", str(self.split / "train_sample_ids.json"),
            "--validation-ids", str(self.split / "validation_sample_ids.json"),
            "--test-ids", str(self.split / "test_sample_ids.json"),
            "--demo-ids", train_ids[0], train_ids[1],
            "--output-dir", str(self.folder / "seed"),
        ])
        self.backend = ScriptedBackend()

    def tearDown(self):
        self.temporary.cleanup()

    def test_seed_creation_and_resume(self):
        state = run(self.args, self.backend)
        self.assertEqual(state["status"], "seeded")
        self.assertEqual(state["version"], 0)
        prompt = self.args.output_dir / "versions/v0000/full_prompt.txt"
        self.assertTrue(prompt.is_file())
        self.assertEqual(len(self.backend.calls), 4)
        call_count = len(self.backend.calls)
        self.assertEqual(run(self.args, self.backend), state)
        self.assertEqual(len(self.backend.calls), call_count)
        self.assertFalse((self.args.output_dir / "windows").exists())
        self.assertFalse((self.args.output_dir / "supervision").exists())

    def test_changed_seed_configuration_is_rejected(self):
        run(self.args, self.backend)
        self.args.teacher_model = "another-model"
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            run(self.args, self.backend)

    def test_heldout_demo_is_rejected_before_api(self):
        self.args.demo_ids[0] = read_json(self.reference / "validation_sample_ids.json")[0]
        with self.assertRaisesRegex(ValueError, "locked Train1600"):
            run(self.args, self.backend)
        self.assertFalse(self.backend.calls)

    def test_dry_run_never_calls_api_or_creates_output(self):
        self.args.dry_run = True
        self.backend.fail_transport = True
        manifest = run(self.args, self.backend)
        self.assertEqual(len(manifest["train_ids"]), 480)
        self.assertFalse(self.args.output_dir.exists())
        self.assertFalse(self.backend.calls)

    def test_only_current_seed_prompts_remain(self):
        for domain in ("metromap", "travelmap"):
            names = {path.name for path in (ROOT / "prompts/initialization" / domain).iterdir()}
            self.assertEqual(names, {"create_skills.txt", "solve_task.txt", "update_usage.txt"})
            original = (ROOT / "prompts" / "student" / domain / "original.txt").read_text()
            frame = (ROOT / "prompts" / "student" / domain / "frame.txt").read_text()
            self.assertEqual(extract_frame(original).rstrip(), frame.rstrip())
            prompt = assemble(frame, fixture_bank(), self.backend.usage, [])
            self.assertEqual(extract_frame(prompt), frame)

    def test_bad_skill_reference_and_metrics_are_rejected(self):
        row = self.backend.rows[self.args.demo_ids[0]]
        value = self.backend.response(row)
        value["output"] = value["output"].replace("[Using Skill 3]", "[Using Skill 99]")
        value["metrics"]["T"] += 1
        checked = verify(value, row, bundle_for(row, "{question}"), 15)
        self.assertFalse(checked["passed"])
        self.assertIn("Unknown skill number", checked["errors"])

    def test_api_backend_uses_image_and_cache_without_network(self):
        backend = APIBackend(timeout=1, max_tokens=128)
        row = self.backend.rows[self.args.demo_ids[0]]
        image = bundle_for(row, "{question}")["image_path"]
        path = self.folder / "api/request.json"
        with patch("rlharness.initialization.execution.openlux_client") as factory:
            client = factory.return_value.__enter__.return_value
            reply = MagicMock()
            reply.choices[0].message.content = '{"how_to_use":"example"}'
            client.chat.completions.create.return_value = reply
            result = backend.request(
                path, model="mock-route", instruction="test",
                payload={"sample_id": "test"}, images=[image],
            )
            self.assertEqual(result, {"how_to_use": "example"})
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            self.assertTrue(any(item["type"] == "image_url" for item in messages[1]["content"]))
            backend.request(
                path, model="mock-route", instruction="test",
                payload={"sample_id": "test"}, images=[image],
            )
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_failed_example_can_be_retried_explicitly(self):
        row = self.backend.rows[self.args.demo_ids[0]]
        prompt = assemble(
            (ROOT / "prompts/student/travelmap/frame.txt").read_text(),
            fixture_bank(), self.backend.usage, [],
        )
        kwargs = {
            "row": row, "prompt": prompt, "skill_count": 15, "model": "mock",
            "template": "test", "folder": self.folder / "solve", "attempts": 1,
        }
        backend = MagicMock()
        backend.request.return_value = {"output": "malformed"}
        self.assertFalse(solve(backend, **kwargs)["accepted"])
        backend.request.return_value = self.backend.response(row)
        self.assertFalse(solve(backend, **kwargs)["accepted"])
        self.assertTrue(solve(backend, **kwargs, retry_failed=True)["accepted"])


if __name__ == "__main__":
    unittest.main()
