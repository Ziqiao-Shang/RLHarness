"""Guards for the immutable Train1600/Val100/Test400 experiment universe."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlharness.common.config import ROOT
from rlharness.data_process import data_split
from rlharness.data_process.skillopt_split import prepare
from rlharness.evolution.skill_pipeline import require_locked_evolution_ids


class FixedProtocolTest(unittest.TestCase):
    def test_resampling_requires_explicit_development_flag(self):
        with tempfile.TemporaryDirectory() as folder, \
             contextlib.redirect_stderr(io.StringIO()), \
             patch.object(sys, "argv", ["data_split", "--output-dir", folder]):
            with self.assertRaises(SystemExit) as raised:
                data_split.main()
        self.assertEqual(raised.exception.code, 2)

    def test_formal_scripts_always_materialize_reference_ids(self):
        for name in (
            "01_generate_sft_data.sh",
            "03_train_rl_round1.sh",
            "04_evolve_skills.sh",
            "05_train_rl_round2.sh",
            "06_evaluate_test.sh",
        ):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("--fixed-ids-dir", text, name)
            self.assertIn("$ROOT/data/reference/metromap/splits", text, name)
            self.assertIn("$ROOT/data/reference/travelmap/splits", text, name)
            self.assertNotIn("FIXED_SPLIT_IDS_DIR", text, name)

    def test_skillopt_rejects_a_replacement_train1600(self):
        reference = ROOT / "data" / "reference" / "metromap" / "splits"
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            output = Path(folder) / "output"
            source.mkdir()
            for name in ("train", "validation", "test"):
                values = json.loads((reference / f"{name}_sample_ids.json").read_text())
                if name == "train":
                    values[0], values[1] = values[1], values[0]
                (source / f"{name}_sample_ids.json").write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "exactly match"):
                prepare("metromap", source, output)

    def test_evolution_rejects_nonreference_order_or_membership(self):
        reference = ROOT / "data" / "reference" / "travelmap" / "splits"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train = json.loads((reference / "train_sample_ids.json").read_text())
            validation = json.loads((reference / "validation_sample_ids.json").read_text())
            train[0], train[1] = train[1], train[0]
            train_path = root / "train.json"
            validation_path = root / "validation.json"
            train_path.write_text(json.dumps(train))
            validation_path.write_text(json.dumps(validation))
            with self.assertRaisesRegex(SystemExit, "exact locked Train1600"):
                require_locked_evolution_ids("travelmap", train_path, validation_path)


if __name__ == "__main__":
    unittest.main()
