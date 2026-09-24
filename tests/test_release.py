"""Offline checks for the public package and command dispatcher."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rlharness.cli import command_for
from rlharness.common.config import ROOT
from rlharness.release import verify_release


class ReleaseTest(unittest.TestCase):
    def test_release_contract(self):
        report = verify_release(ROOT)
        self.assertEqual(report["errors"], [])

    def test_dispatcher_keeps_domains_explicit(self):
        command = command_for("rl-round2", "travelmap", ["trainer.save_freq=50"])
        self.assertEqual(command[:2], ["bash", str(ROOT / "scripts/05_train_rl_round2.sh")])
        self.assertEqual(command[2:4], ["--domain", "travelmap"])
        self.assertEqual(command[-1], "trainer.save_freq=50")

    def test_training_configs_are_flat_and_shared(self):
        expected = {
            "skillopt.yaml",
            "sft_train.yaml",
            "sft_merge.yaml",
            "rl_round1.yaml",
            "rl_round2.yaml",
        }
        self.assertTrue(expected.issubset({path.name for path in (ROOT / "configs").glob("*.yaml")}))
        for redundant in ("common", "metromap", "travelmap"):
            self.assertFalse((ROOT / "configs" / redundant).exists())

        round1 = (ROOT / "scripts/03_train_rl_round1.sh").read_text(encoding="utf-8")
        round2 = (ROOT / "scripts/05_train_rl_round2.sh").read_text(encoding="utf-8")
        self.assertIn("$ROOT/configs/rl_round1.yaml", round1)
        self.assertIn("$ROOT/configs/rl_round2.yaml", round2)
        self.assertNotIn("configs/$DOMAIN", round1 + round2)

    def test_sft_registry_is_generated_outside_vendor_source(self):
        script = (ROOT / "scripts/02_train_sft.sh").read_text(encoding="utf-8")
        self.assertIn('REGISTRY_DIR="${SFT_REGISTRY_DIR:-$DATA_DIR/llamafactory}"', script)
        self.assertIn('"$REGISTRY_DIR/dataset_info.json"', script)
        self.assertIn('"dataset_dir=$REGISTRY_DIR"', script)
        self.assertNotIn("third_party/LlamaFactory/data/dataset_info.json", script)

    def test_sft_teacher_defaults_to_gpt56_sol(self):
        script = (ROOT / "scripts/01_generate_sft_data.sh").read_text(encoding="utf-8")
        self.assertIn('MODEL="${TEACHER_MODEL:-gpt-5.6-sol}"', script)
        self.assertNotIn("gemini-3.7-flash", script)

    def test_sft_data_stage_uses_declared_lightweight_tokenizer(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        teacher = (
            ROOT / "src/rlharness/data_process/sft_teacher.py"
        ).read_text(encoding="utf-8")
        builder = (
            ROOT / "src/rlharness/data_process/sft_build.py"
        ).read_text(encoding="utf-8")

        self.assertIn("tokenizers>=0.22,<0.23", pyproject)
        self.assertIn("tokenizers>=0.22,<0.23", requirements)
        self.assertNotIn("from transformers", teacher)
        self.assertNotIn("from transformers", builder)

    def test_sft_registry_runtime_does_not_touch_vendor_source(self):
        vendor_registry = ROOT / "third_party/LlamaFactory/data/dataset_info.json"
        before = vendor_registry.read_bytes() if vendor_registry.is_file() else None
        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder)
            generated = temporary / "generated"
            data = generated / "sft"
            base = temporary / "base"
            output = temporary / "output"
            merged = temporary / "merged"
            registry = temporary / "registry"
            data.mkdir(parents=True)
            base.mkdir()
            (data / "train.json").write_text("[]\n", encoding="utf-8")
            (base / "config.json").write_text("{}\n", encoding="utf-8")
            fake_cli = temporary / "llamafactory-cli"
            fake_cli.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "mode=\"$1\"\n"
                "shift\n"
                "for arg in \"$@\"; do\n"
                "  case \"$arg\" in\n"
                "    output_dir=*) output=${arg#output_dir=} ;;\n"
                "    export_dir=*) export_dir=${arg#export_dir=} ;;\n"
                "  esac\n"
                "done\n"
                "if [[ \"$mode\" == train ]]; then\n"
                "  mkdir -p \"$output/checkpoint-1\"\n"
                "  printf '{}\\n' > \"$output/checkpoint-1/adapter_config.json\"\n"
                "else\n"
                "  mkdir -p \"$export_dir\"\n"
                "  printf '{}\\n' > \"$export_dir/config.json\"\n"
                "fi\n",
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)
            env = dict(os.environ)
            env.update(
                QWEN35_MODEL=str(base),
                GENERATED_DATA_DIR=str(generated),
                SFT_REGISTRY_DIR=str(registry),
                SFT_OUTPUT_DIR=str(output),
                SFT_MERGED_MODEL=str(merged),
                SELECTED_SFT_CHECKPOINT_FILE=str(temporary / "selected.txt"),
                LLAMAFACTORY_CLI=str(fake_cli),
                PYTHON_BIN=sys.executable,
                PYTHONDONTWRITEBYTECODE="1",
            )
            subprocess.run(
                ["bash", str(ROOT / "scripts/02_train_sft.sh"), "--domain", "metromap"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            info = json.loads((registry / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(set(info), {"metromap_skill_sft"})
            self.assertEqual(
                info["metromap_skill_sft"]["file_name"], str((data / "train.json").resolve())
            )
            self.assertTrue((merged / "config.json").is_file())

        after = vendor_registry.read_bytes() if vendor_registry.is_file() else None
        self.assertEqual(after, before)

    def test_only_current_initialization_flow_is_shipped(self):
        required = (
            ROOT / "src/rlharness/initialization/seed.py",
            ROOT / "src/rlharness/initialization/skillopt_run.py",
            ROOT / "src/skillopt/engine/trainer.py",
        )
        removed = (
            ROOT / "src/rlharness/initialization/pipeline.py",
            ROOT / "src/rlharness/initialization/handoff.py",
            ROOT / "prompts/initialization/metromap/update_skills.txt",
            ROOT / "prompts/initialization/travelmap/update_skills.txt",
        )
        self.assertTrue(all(path.is_file() for path in required))
        self.assertTrue(all(not path.exists() for path in removed))

    def test_default_evaluation_consumes_round_two_output(self):
        script = (ROOT / "scripts/06_evaluate_test.sh").read_text(encoding="utf-8")
        self.assertIn('DEFAULT_MODEL="$ROOT/artifacts/sft_merged"', script)
        self.assertIn('DEFAULT_ROUND2_OUTPUT="$ROOT/artifacts/grpo_round2"', script)
        self.assertIn('DEFAULT_MODEL="$ROOT/artifacts/travelmap/sft_merged"', script)
        self.assertIn('DEFAULT_ROUND2_OUTPUT="$ROOT/artifacts/travelmap/grpo_round2"', script)
        self.assertIn('PROMPT="$ROUND2_OUTPUT/selected_prompt.txt"', script)
        self.assertIn("verify-round2", script)
        self.assertNotIn("../metromap_final", script)
        self.assertNotIn("../travelmap_final", script)

    def test_round_two_locks_selected_prompt(self):
        script = (ROOT / "scripts/05_train_rl_round2.sh").read_text(encoding="utf-8")
        self.assertIn('PROMPT="$SELECTION_DIR/full_prompt.txt"', script)
        self.assertIn('SELECTION_MANIFEST="$SELECTION_DIR/selection.json"', script)
        self.assertIn("lock-round2", script)
        self.assertIn('PROMPT="$OUTPUT/selected_prompt.txt"', script)

    def test_round_one_locks_original_prompt(self):
        script = (ROOT / "scripts/03_train_rl_round1.sh").read_text(encoding="utf-8")
        self.assertIn("lock-round1", script)
        self.assertIn('PROMPT="$OUTPUT/selected_prompt.txt"', script)


if __name__ == "__main__":
    unittest.main()
