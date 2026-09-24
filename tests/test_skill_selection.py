"""Deterministic tests for candidate selection and round-two Prompt locking."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlharness.evolution.skill_pipeline import CANDIDATE_NAMES
from rlharness.evolution.skill_select import (
    auto_select_candidate,
    lock_round1_prompt,
    lock_round2_prompt,
    select_candidate,
    verify_round2_prompt,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class SkillSelectionTest(unittest.TestCase):
    def build_run(self, root: Path) -> Path:
        run = root / "skill_evolution"
        review_rows = []
        for index, name in enumerate(CANDIDATE_NAMES):
            candidate = run / "candidates" / name
            evaluation = run / "validation_qwen35plus" / name
            candidate.mkdir(parents=True)
            evaluation.mkdir(parents=True)
            prompt = candidate / "full_prompt.txt"
            prompt.write_text(f"full prompt for {name}\n", encoding="utf-8")
            evaluated_prompt = evaluation / "evaluated_prompt.txt"
            evaluated_prompt.write_bytes(prompt.read_bytes())
            (candidate / "planning_skills.txt").write_text(
                "1. Inspect the map\n", encoding="utf-8"
            )
            write_json(candidate / "candidate.json", {"candidate_name": name, "skills": []})
            write_json(
                candidate / "fewshot_rewrite.validation.json",
                {"valid": True, "required_procedure_preserved": True},
            )
            write_json(
                evaluation / "evaluation_manifest.json",
                {
                    "candidate": name,
                    "domain": "metromap",
                    "prompt": str(prompt.resolve()),
                    "prompt_snapshot": str(evaluated_prompt.resolve()),
                },
            )
            write_json(
                evaluation / "predictions.summary.json",
                {
                    "count": 100,
                    "hard_correct": 60 + index,
                    "all_acc": 0.0,
                    "part_acc": 0.0,
                    "format_rate": 0.0,
                    "truncated": 0,
                },
            )
            (evaluation / "predictions.jsonl").write_text(
                json.dumps({"sample_id": f"sample-{index}", "prediction": "answer"}) + "\n",
                encoding="utf-8",
            )
            review_rows.append(
                {
                    "candidate": name,
                    "count": 100,
                    "hard_correct": 60 + index,
                    "exact_accuracy": (60 + index) / 100,
                    "all_acc": 0.0,
                    "part_acc": 0.0,
                    "format_rate": 0.0,
                    "truncated": 0,
                    "skill_usage": {"per_skill": [{"skill_id": 1, "usage_rate": 0.5}]},
                }
            )
        write_json(
            run / "validation_qwen35plus/candidate_review.json",
            {"decision_mode": "gpt-5.6-sol", "selected_candidate": None, "candidates": review_rows},
        )
        return run

    def test_selection_and_round_two_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = self.build_run(root)
            selected = run / "selected"
            selection_path = select_candidate(
                domain="metromap",
                candidate="candidate_b_stage_based",
                run_dir=run,
                output_dir=selected,
            )
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            self.assertEqual(selection["candidate"], "candidate_b_stage_based")
            self.assertEqual(selection["metrics"]["hard_correct"], 61)
            self.assertIn("sources", selection)

            # Repeating the same explicit override is idempotent.
            select_candidate(
                domain="metromap",
                candidate="candidate_b_stage_based",
                run_dir=run,
                output_dir=selected,
            )

            round2 = root / "grpo_round2"
            snapshot = lock_round2_prompt(
                domain="metromap",
                prompt=selected / "full_prompt.txt",
                output_dir=round2,
                selection_manifest=selection_path,
            )
            self.assertEqual(snapshot, verify_round2_prompt(domain="metromap", output_dir=round2))
            self.assertEqual(snapshot.read_bytes(), (selected / "full_prompt.txt").read_bytes())

    def test_selection_rejects_unreviewed_prompt_or_silent_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = self.build_run(root)
            selected = run / "selected"
            select_candidate(
                domain="metromap",
                candidate="candidate_a_conservative",
                run_dir=run,
                output_dir=selected,
            )
            with self.assertRaises(SystemExit):
                select_candidate(
                    domain="metromap",
                    candidate="candidate_c_compact",
                    run_dir=run,
                    output_dir=selected,
                )

            manifest = run / "validation_qwen35plus/candidate_c_compact/evaluation_manifest.json"
            write_json(manifest, {"candidate": "candidate_c_compact", "domain": "metromap", "prompt": "/tmp/other.txt"})
            with self.assertRaises(SystemExit):
                select_candidate(
                    domain="metromap",
                    candidate="candidate_c_compact",
                    run_dir=run,
                    output_dir=selected,
                    replace=True,
                )

    def test_round_two_output_cannot_switch_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("first\n", encoding="utf-8")
            second.write_text("second\n", encoding="utf-8")
            output = root / "round2"
            lock_round2_prompt(domain="travelmap", prompt=first, output_dir=output)
            with self.assertRaises(SystemExit):
                lock_round2_prompt(domain="travelmap", prompt=second, output_dir=output)

    def test_round_one_output_cannot_switch_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prompt = root / "original.txt"
            prompt.write_text("first version\n", encoding="utf-8")
            output = root / "round1"
            snapshot = lock_round1_prompt(
                domain="metromap", prompt=prompt, output_dir=output
            )
            self.assertEqual(snapshot.read_bytes(), b"first version\n")

            lock_round1_prompt(domain="metromap", prompt=prompt, output_dir=output)
            prompt.write_text("changed version\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                lock_round1_prompt(domain="metromap", prompt=prompt, output_dir=output)

    def test_selection_rejects_prompt_changed_after_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.build_run(Path(folder))
            prompt = run / "candidates/candidate_a_conservative/full_prompt.txt"
            prompt.write_text("changed after Val100\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "content differs"):
                select_candidate(
                    domain="metromap",
                    candidate="candidate_a_conservative",
                    run_dir=run,
                    output_dir=run / "selected",
                )

    def test_gpt56_selector_uses_existing_review_and_records_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = self.build_run(root)
            template = root / "select.txt"
            template.write_text("Review:\n{candidate_review}\nReturn JSON.", encoding="utf-8")
            response = json.dumps(
                {
                    "selected_candidate": "candidate_c_compact",
                    "reason": "It has the strongest exact accuracy with the same observed Skill use.",
                }
            )
            with mock.patch(
                "rlharness.evolution.skill_select.chat_text",
                return_value=response,
            ) as call:
                selection_path = auto_select_candidate(
                    domain="metromap",
                    run_dir=run,
                    output_dir=run / "selected",
                    template_path=template,
                    client=object(),
                )
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            self.assertEqual(selection["candidate"], "candidate_c_compact")
            self.assertEqual(selection["selection_mode"], "gpt56sol_automatic")
            self.assertEqual(selection["selection_model"], "gpt-5.6-sol")
            self.assertIn("strongest exact accuracy", selection["reason"])
            self.assertEqual(call.call_args.args[1], "gpt-5.6-sol")

            # Completed automatic selections resume without another API call.
            with mock.patch(
                "rlharness.evolution.skill_select.chat_text",
                side_effect=AssertionError("selector should not be called again"),
            ):
                self.assertEqual(
                    auto_select_candidate(
                        domain="metromap",
                        run_dir=run,
                        output_dir=run / "selected",
                        template_path=template,
                        client=object(),
                    ),
                    selection_path,
                )

    def test_gpt56_selector_rejects_unknown_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = self.build_run(root)
            template = root / "select.txt"
            template.write_text("{candidate_review}", encoding="utf-8")
            with mock.patch(
                "rlharness.evolution.skill_select.chat_text",
                return_value='{"selected_candidate":"candidate_d","reason":"unsupported"}',
            ):
                with self.assertRaises(SystemExit):
                    auto_select_candidate(
                        domain="metromap",
                        run_dir=run,
                        output_dir=run / "selected",
                        template_path=template,
                        client=object(),
                    )


if __name__ == "__main__":
    unittest.main()
