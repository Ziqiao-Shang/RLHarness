"""Offline integration tests; model responses are synthetic, never API calls."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlharness.common.config import ROOT
from rlharness.data_process.skillopt_split import prepare
from rlharness.data_process.data_split import materialize_fixed
from rlharness.initialization.skillopt_run import (
    LockedMapLoader, MapSkillOptAdapter, ReflACTTrainer, flatten_config, load_config, lock_run,
    parse_args, run,
)
from rlharness.initialization.state import read_json, write_json
from skillopt.engine.trainer import _redact_cfg
from skillopt.evaluation.gate import evaluate_gate
from skillopt.model import azure_openai
from test_initialization import ScriptedBackend


class SkillOptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.split = Path(cls.temp.name) / "splits"
        prepare("travelmap", ROOT / "data/reference/travelmap/splits", cls.split)
        cls.loader = LockedMapLoader("travelmap", cls.split)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def config(self, output):
        cfg = flatten_config(load_config(str(ROOT / "configs/skillopt.yaml")))
        cfg.update(env="travelmap", out_root=str(output),
                   skill_init=str(ROOT / "prompts/student/travelmap/original.txt"))
        return cfg

    def test_locked_plan_and_no_test_access(self):
        plan = self.loader.plan_train_epoch(epoch=1, steps_per_epoch=12,
                                            accumulation=1, batch_size=40, seed=42)
        ids = [r["id"] for b in plan for r in b.payload]
        self.assertEqual(len(set(ids)), 480)
        self.assertEqual(ids, [s for b in self.loader.batches for s in b])
        with self.assertRaises(ValueError):
            self.loader.build_eval_batch(400, "valid_unseen", 42)
        self.assertEqual(self.loader.build_eval_batch(100, "valid_seen", 123).payload,
                         self.loader.val_items)

    def test_guard_and_config_lock(self):
        adapter = MapSkillOptAdapter("travelmap", self.loader)
        adapter.setup(self.config(Path(self.temp.name) / "unused"))
        self.assertTrue(adapter.validate_candidate_skill(adapter.baseline)[0])
        self.assertFalse(adapter.validate_candidate_skill(adapter.baseline.replace("{w1}", "{x}"))[0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            lock_run(path, {"model": "A"})
            lock_run(path, {"model": "A"})
            with self.assertRaises(ValueError):
                lock_run(path, {"model": "B"})

    def test_tie_is_rejected(self):
        decision = evaluate_gate("new", .5, "old", .5, "old", .5, 0, 1)
        self.assertEqual(decision.action, "reject")

    def test_student_rollout_cache_and_no_gt_input(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self.config(Path(folder))
            adapter = MapSkillOptAdapter("travelmap", self.loader)
            adapter.setup(cfg)
            row = self.loader.train_items[0]
            reply = f"<reasoning>Reasoning.</reasoning><response>{row['gt_route']}</response>"
            with patch("rlharness.initialization.skillopt_adapter.chat_target_messages",
                       return_value=(reply, {})) as chat:
                results = adapter.rollout([row], adapter.baseline, folder)
                self.assertEqual(results[0]["hard"], 1)
                messages = chat.call_args.args[0]
                serialized = json.dumps(messages)
                self.assertNotIn("Ground-truth route", serialized)
                self.assertNotIn("GT route:", serialized)
                self.assertIn("image_url", serialized)
                adapter.rollout([row], adapter.baseline, folder)
                self.assertEqual(chat.call_count, 1)
            with patch("rlharness.initialization.skillopt_adapter.chat_target_messages",
                       side_effect=RuntimeError("simulated API failure")):
                with self.assertRaises(RuntimeError):
                    adapter.rollout([self.loader.train_items[1]], adapter.baseline, folder)

    def test_teacher_empty_edits_valid_but_missing_response_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = MapSkillOptAdapter("travelmap", self.loader)
            adapter.setup(self.config(Path(folder)))
            row = dict(self.loader.train_items[0], hard=0, task_type="travelmap")
            write_json(Path(folder) / "predictions" / row["id"] / "conversation.json",
                       [{"role": "assistant", "content": "synthetic response"}])
            with patch("skillopt.gradient.reflect.chat_optimizer",
                       return_value=('{"patch":{"edits":[]}}', {})):
                self.assertEqual(len(adapter.reflect([row], adapter.baseline, folder)), 1)
        with tempfile.TemporaryDirectory() as folder:
            with patch("skillopt.gradient.reflect.chat_optimizer",
                       return_value=("invalid JSON", {})):
                with self.assertRaises(RuntimeError):
                    adapter.reflect([row], adapter.baseline, folder)

    def test_teacher_invalid_json_is_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = MapSkillOptAdapter("travelmap", self.loader)
            adapter.setup(self.config(Path(folder)))
            row = dict(self.loader.train_items[0], hard=0, task_type="travelmap")
            write_json(Path(folder) / "predictions" / row["id"] / "conversation.json",
                       [{"role": "assistant", "content": "synthetic response"}])
            responses = [("invalid JSON", {}), ('{"patch":{"edits":[]}}', {})]
            with patch("skillopt.gradient.reflect.chat_optimizer",
                       side_effect=responses) as chat:
                patches = adapter.reflect([row], adapter.baseline, folder)
                self.assertEqual(len(patches), 1)
                self.assertEqual(chat.call_count, 2)

    def test_models_do_not_share_gpt_reasoning_parameter(self):
        calls = []
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (calls.append(kwargs), response)[1])))
        with patch.object(azure_openai, "REASONING_EFFORT", "medium"):
            for model in ("gemini-3.5-flash", "gpt-5.6-sol"):
                azure_openai._chat_messages_impl(client, model, [], 32, 1, "test")
        self.assertNotIn("reasoning_effort", calls[0])
        self.assertEqual(calls[1]["reasoning_effort"], "medium")

    def test_secret_redaction(self):
        self.assertEqual(_redact_cfg({"optimizer_azure_openai_api_key": "secret"}),
                         {"optimizer_azure_openai_api_key": "[REDACTED]"})

    def test_sft_handoff_preserves_locked_ids(self):
        for domain in ("metromap", "travelmap"):
            source = ROOT / "data/reference" / domain / "splits"
            with tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                manifest = materialize_fixed(domain, source, output)
                for name in ("train", "validation", "test"):
                    self.assertEqual(read_json(output / f"{name}_sample_ids.json"),
                                     read_json(source / f"{name}_sample_ids.json"))
                self.assertEqual(manifest["train_rows"], 1600)
                write_json(output / "train_sample_ids.json", ["wrong"])
                with self.assertRaises(ValueError):
                    materialize_fixed(domain, source, output)

    def test_seed_then_skillopt_both_domains_and_resume(self):
        for domain in ("metromap", "travelmap"):
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                split = base / "splits"
                prepare(domain, ROOT / "data/reference" / domain / "splits", split)
                args = parse_args(["--domain", domain, "--split-dir", str(split),
                                   "--output-dir", str(base / "run")])
                backend = ScriptedBackend(domain)
                seen = []

                class FakeTrainer:
                    interrupt = True

                    def __init__(self, cfg, adapter):
                        self.cfg, self.adapter = cfg, adapter

                    def train(self):
                        prompt = Path(self.cfg["skill_init"]).read_text()
                        seen.append(prompt)
                        assert "Fixture skill 1" in prompt
                        assert "[Using Skill 1]" in prompt
                        assert self.cfg["max_train_steps"] == 12
                        assert self.cfg["target_model"] == "gemini-3.5-flash"
                        assert self.cfg["optimizer_model"] == "gpt-5.6-sol"
                        assert self.adapter.validate_candidate_skill(prompt)[0]
                        if self.interrupt:
                            raise RuntimeError("optimizer interrupted after seed")
                        (Path(self.cfg["out_root"]) / "best_skill.md").write_text(prompt)
                        return {"best_step": 0}

                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "optimizer interrupted"):
                        run(args, seed_backend=backend, trainer_cls=FakeTrainer)
                    count = len(backend.calls)
                    self.assertEqual(count, 4)  # bank, usage, two examples
                    self.assertFalse((base / "run/best/full_prompt.txt").exists())
                    FakeTrainer.interrupt = False
                    run(args, seed_backend=backend, trainer_cls=FakeTrainer)
                self.assertEqual(len(backend.calls), count)
                self.assertEqual(seen[0], seen[1])
                self.assertEqual((base / "run/best/full_prompt.txt").read_text(), seen[0])
                self.assertFalse((base / "run/seed/windows").exists())
                self.assertFalse((base / "run/seed/supervision").exists())
                manifest = read_json(base / "run/seed/manifest.json")
                if domain == "metromap":
                    self.assertGreater(manifest["training_demo_heldout_figure_overlap"], 0)
                args.demo_ids = [args_id for args_id in read_json(split / "train_sample_ids.json")[2:4]]
                with self.assertRaisesRegex(ValueError, "changed"):
                    run(args, seed_backend=backend, trainer_cls=FakeTrainer)

    def test_combined_dry_run_never_creates_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parse_args(["--domain", "travelmap", "--split-dir", str(self.split),
                               "--output-dir", str(Path(folder) / "run"), "--dry-run"])
            backend = ScriptedBackend()
            with contextlib.redirect_stdout(io.StringIO()), \
                 patch("rlharness.initialization.seed.APIBackend",
                       side_effect=AssertionError("No API client in dry-run")):
                record = run(args, seed_backend=backend)
            self.assertFalse(args.output_dir.exists())
            self.assertFalse(backend.calls)
            self.assertEqual(len(record["seed_initialization"]["demo_ids"]), 2)

    def test_provided_prompt_skips_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            prompt = ROOT / "prompts/student/travelmap/original.txt"
            args = parse_args(["--domain", "travelmap", "--split-dir", str(self.split),
                               "--output-dir", str(Path(folder) / "run"),
                               "--initial-prompt", str(prompt), "--dry-run"])
            with contextlib.redirect_stdout(io.StringIO()), \
                 patch("rlharness.initialization.seed.prepare",
                       side_effect=AssertionError("Seed creation must be bypassed")):
                record = run(args)
            self.assertEqual(record["seed_initialization"]["mode"], "provided_prompt")

    def test_single_step_override(self):
        with tempfile.TemporaryDirectory() as folder:
            prompt = ROOT / "prompts/student/travelmap/original.txt"
            args = parse_args([
                "--domain", "travelmap", "--split-dir", str(self.split),
                "--output-dir", str(Path(folder) / "run"),
                "--initial-prompt", str(prompt), "--max-train-steps", "1", "--dry-run",
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                record = run(args)
            self.assertEqual(record["config"]["max_train_steps"], 1)

    def test_twelve_steps_interrupt_resume_accept_reject(self):
        loader = self.loader
        observed = []
        contexts = []

        class SyntheticAdapter(MapSkillOptAdapter):
            stop_at = 4

            def rollout(self, rows, skill, out_dir, **kwargs):
                if "steps" in Path(out_dir).parts and "rollout" == Path(out_dir).name:
                    step = int(Path(out_dir).parent.name.split("_")[-1])
                    if step == self.stop_at:
                        raise RuntimeError("simulated interruption")
                    observed.extend(r["id"] for r in rows)
                quality = 60 if "SYNTHETIC_GOOD" in skill else 50
                if "SYNTHETIC_BAD" in skill:
                    quality = 40
                return [{"id": r["id"], "hard": int(i < quality), "soft": int(i < quality),
                         "task_type": "travelmap", "fail_reason": "synthetic"}
                        for i, r in enumerate(rows)]

            def reflect(self, results, skill, out_dir, **kwargs):
                contexts.append(kwargs.get("step_buffer_context", ""))
                line = next(s for s in skill.splitlines() if s.startswith("1. "))
                step = int(Path(out_dir).name.split("_")[-1])
                suffix = " SYNTHETIC_GOOD" if step == 1 else " SYNTHETIC_BAD"
                return [{"source_type": "failure", "patch": {"edits": [
                    {"op": "replace", "target": line, "content": line + suffix}]}}]

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            cfg = self.config(Path(folder))
            adapter = SyntheticAdapter("travelmap", loader)
            with patch("skillopt.engine.trainer.merge_patches", side_effect=lambda s, f, u, **k: f[0]), \
                 patch("skillopt.engine.trainer.rank_and_select", side_effect=lambda s, p, **k: p), \
                 patch("skillopt.model.azure_openai.OpenAI", side_effect=AssertionError("No live API")):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    ReflACTTrainer(cfg.copy(), adapter).train()
                self.assertEqual(read_json(Path(folder) / "runtime_state.json")["last_completed_step"], 3)
                adapter.stop_at = 0
                summary = ReflACTTrainer(cfg.copy(), adapter).train()
            self.assertEqual(len(observed), 480)
            self.assertEqual(len(set(observed)), 480)
            self.assertEqual(summary["total_steps"], 12)
            self.assertEqual(summary["total_accepts"], 1)
            self.assertEqual(summary["total_rejects"], 11)
            self.assertEqual(summary["best_step"], 1)
            self.assertIn("SYNTHETIC_BAD", contexts[3])
            best = (Path(folder) / "best_skill.md").read_text()
            self.assertIn("SYNTHETIC_GOOD", best)
            self.assertNotIn("SYNTHETIC_BAD", best)
            self.assertIsNone(summary["test_hard"])


if __name__ == "__main__":
    unittest.main()
