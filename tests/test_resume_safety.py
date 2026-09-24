"""Offline regression tests for resumable SFT and Skill evolution outputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlharness.data_process.sft_teacher import lock_generation_manifest
from rlharness.evolution.skill_generate import (
    candidate_is_complete,
    candidate_text,
    materialize_full_prompt,
)
from rlharness.evolution.skill_pipeline import lock_evaluated_prompt


class ResumeSafetyTest(unittest.TestCase):
    def test_sft_resume_requires_identical_generation_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "labels.manifest.json"
            output = root / "labels.jsonl"
            payload = {
                "model": "gpt-5.6-sol",
                "student_prompt": "prompt one",
                "teacher_prompt": "teacher one",
                "selection": {"sample_ids": ["a", "b"]},
            }
            lock_generation_manifest(
                manifest, payload, artifacts=(output,), resume=True
            )
            output.write_text('{"sample_id":"a"}\n', encoding="utf-8")
            lock_generation_manifest(
                manifest, payload, artifacts=(output,), resume=True
            )

            changed = {**payload, "model": "another-model"}
            with self.assertRaisesRegex(SystemExit, "inputs differ"):
                lock_generation_manifest(
                    manifest, changed, artifacts=(output,), resume=True
                )

    def test_sft_resume_rejects_outputs_without_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "labels.jsonl"
            output.write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "no generation manifest"):
                lock_generation_manifest(
                    root / "labels.manifest.json",
                    {"model": "gpt-5.6-sol"},
                    artifacts=(output,),
                    resume=True,
                )

    def test_val_predictions_are_bound_to_exact_prompt_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prompt = root / "candidate/full_prompt.txt"
            output = root / "validation/predictions.jsonl"
            prompt.parent.mkdir(parents=True)
            output.parent.mkdir(parents=True)
            prompt.write_text("candidate prompt\n", encoding="utf-8")
            snapshot = lock_evaluated_prompt(prompt, output)
            self.assertEqual(snapshot.read_bytes(), prompt.read_bytes())

            output.write_text("prediction\n", encoding="utf-8")
            lock_evaluated_prompt(prompt, output)
            prompt.write_text("regenerated prompt\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "differs"):
                lock_evaluated_prompt(prompt, output)

    def test_generated_candidate_cache_checks_all_materialized_files(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate_dir = Path(folder) / "candidate_a_conservative"
            candidate_dir.mkdir()
            source_prompt = (
                "# Task\n\n"
                "# Planning Skills\n\n1. Old skill\n\n"
                "# How to Use the Skills\nUse them.\n"
            )
            candidate = {
                "candidate_name": "candidate_a_conservative",
                "skills": [
                    {
                        "title": "Trace Edges",
                        "trigger": "When building a route.",
                        "action": "Trace each visible edge.",
                        "guard": "Do not infer adjacency from table order.",
                        "check": "Every pair has a visible edge.",
                    }
                ],
                "operation_log": [
                    {
                        "operation": "KEEP",
                        "source_skill_numbers": [1],
                        "result_titles": ["Trace Edges"],
                        "evidence": "Repeated route traces support this behavior.",
                    }
                ],
                "coverage_audit": [
                    {
                        "source_skill_number": 1,
                        "core_responsibilities": ["Trace visible map edges."],
                        "disposition": "PRESERVED",
                        "covered_by_result_titles": ["Trace Edges"],
                        "deletion_evidence": [],
                    }
                ],
                "generation": {"model": "gpt-5.6-sol", "attempt": 1},
            }
            skills = candidate_text(candidate)
            (candidate_dir / "candidate.json").write_text(
                json.dumps(candidate) + "\n", encoding="utf-8"
            )
            (candidate_dir / "planning_skills.txt").write_text(skills, encoding="utf-8")
            pre_rewrite = materialize_full_prompt(source_prompt, skills)
            (candidate_dir / "pre_rewrite_prompt.txt").write_text(
                pre_rewrite, encoding="utf-8"
            )

            self.assertTrue(
                candidate_is_complete(
                    candidate_dir,
                    name="candidate_a_conservative",
                    model="gpt-5.6-sol",
                    source_skill_ids={1},
                    source_prompt=source_prompt,
                )
            )
            (candidate_dir / "pre_rewrite_prompt.txt").write_text(
                "stale prompt\n", encoding="utf-8"
            )
            self.assertFalse(
                candidate_is_complete(
                    candidate_dir,
                    name="candidate_a_conservative",
                    model="gpt-5.6-sol",
                    source_skill_ids={1},
                    source_prompt=source_prompt,
                )
            )


if __name__ == "__main__":
    unittest.main()
