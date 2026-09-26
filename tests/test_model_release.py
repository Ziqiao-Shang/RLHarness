import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlharness import model_release
from rlharness.eval import eval_local


ROOT = Path(__file__).resolve().parents[1]


class ModelReleaseTest(unittest.TestCase):
    def test_public_release_coordinates_are_explicit(self):
        self.assertEqual(
            model_release.DEFAULT_MODEL_REPO,
            "szq-nju/RLHarness-MapTab-Models",
        )
        self.assertEqual(
            model_release.DEFAULT_MODEL_SUBFOLDERS,
            {
                "metromap": "metromap_trained",
                "travelmap": "travelmap_trained",
            },
        )

    def test_pretrained_entrypoint_uses_domain_model_without_adapter(self):
        script = (ROOT / "scripts" / "07_evaluate_pretrained.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("python", script.splitlines()[0].lower())
        self.assertIn("-m rlharness.model_release", script)
        self.assertIn("--print-local-path", script)
        self.assertIn('--local-path "$MODEL_PATH"', script)
        self.assertIn('MODEL_PATH="$RESOLVED_MODEL_PATH"', script)
        self.assertIn("--check-only", script)
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
            "RLHARNESS_MODEL_SUBFOLDER",
            "RLHARNESS_LOCAL_MODEL_PATH",
        ):
            self.assertIn(name, source)

    def test_local_model_path_validates_without_hub_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5"}),
                encoding="utf-8",
            )
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer": "model-00001-of-00001.safetensors"
                        }
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "model-00001-of-00001.safetensors").write_bytes(
                b"weights"
            )
            with mock.patch("rlharness.model_release.HfApi") as api_class:
                resolution = model_release.validate_local_model(
                    domain="travelmap",
                    model_path=model_dir,
                )
            api_class.assert_not_called()
            self.assertEqual(resolution["model"]["source"], "local")
            self.assertEqual(
                resolution["model"]["weight_shards"],
                ["model-00001-of-00001.safetensors"],
            )

    def test_qwen35_uses_direct_transformers_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                eval_local.resolve_eval_backend(model_dir, "auto"),
                "transformers",
            )
            self.assertEqual(
                eval_local.resolve_eval_backend(model_dir, "vllm"),
                "vllm",
            )
            with mock.patch(
                "rlharness.eval.eval_local.importlib.metadata.version",
                return_value="0.26.0",
            ):
                self.assertEqual(
                    eval_local.resolve_eval_backend(model_dir, "auto"),
                    "vllm",
                )

    @mock.patch("rlharness.model_release.snapshot_download")
    @mock.patch("rlharness.model_release.HfApi")
    def test_download_uses_published_subfolder_and_validates_shards(
        self, api_class, snapshot_download
    ):
        subfolder = "travelmap_trained"
        filenames = [
            f"{subfolder}/config.json",
            f"{subfolder}/model.safetensors.index.json",
            f"{subfolder}/model-00001-of-00001.safetensors",
        ]
        api_class.return_value.model_info.return_value = SimpleNamespace(
            sha="resolved-commit",
            siblings=[SimpleNamespace(rfilename=name) for name in filenames],
        )

        def materialize(**kwargs):
            model_dir = Path(kwargs["local_dir"]) / subfolder
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5"}) + "\n",
                encoding="utf-8",
            )
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")

        snapshot_download.side_effect = materialize
        with tempfile.TemporaryDirectory() as tmp:
            resolution = model_release.download_released_model(
                domain="travelmap",
                output_dir=Path(tmp),
            )

            self.assertEqual(resolution["model"]["subfolder"], subfolder)
            self.assertEqual(resolution["model"]["resolved_commit"], "resolved-commit")
            self.assertEqual(
                Path(resolution["model"]["local_path"]), Path(tmp) / subfolder
            )
            self.assertTrue((Path(tmp) / "travelmap_resolution.json").is_file())
        self.assertEqual(
            snapshot_download.call_args.kwargs["allow_patterns"],
            ["travelmap_trained/*"],
        )


if __name__ == "__main__":
    unittest.main()
