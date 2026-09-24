import unittest
from pathlib import Path

from rlharness import model_release


ROOT = Path(__file__).resolve().parents[1]


class ModelReleaseTest(unittest.TestCase):
    def test_public_release_coordinates_are_explicit(self):
        self.assertEqual(
            model_release.DEFAULT_MODEL_REPO,
            "szq-nju/RLHarness-MapTab-Models",
        )

    def test_pretrained_entrypoint_uses_domain_model_without_adapter(self):
        script = (ROOT / "scripts" / "07_evaluate_pretrained.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("python", script.splitlines()[0].lower())
        self.assertIn("-m rlharness.model_release", script)
        self.assertIn('MODEL_PATH="$RELEASE_ROOT/$DOMAIN"', script)
        self.assertIn("env -u ADAPTER_PATH", script)
        self.assertIn("scripts/06_evaluate_test.sh", script)

    def test_download_interface_keeps_model_coordinates_overridable(self):
        source = (ROOT / "src" / "rlharness" / "model_release.py").read_text(
            encoding="utf-8"
        )
        for name in (
            "RLHARNESS_MODEL_ROOT",
            "RLHARNESS_MODEL_REPO",
            "RLHARNESS_MODEL_REVISION",
        ):
            self.assertIn(name, source)


if __name__ == "__main__":
    unittest.main()
