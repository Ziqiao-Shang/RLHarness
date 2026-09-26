import unittest
from pathlib import Path

from rlharness.data_process import prepare_data


class PrepareDataTest(unittest.TestCase):
    def test_official_release_is_pinned_and_complete(self):
        self.assertEqual(
            prepare_data.PUBLIC_DATASET_REPO,
            "szq-nju/MapTab",
        )
        self.assertRegex(prepare_data.PUBLIC_REVISION, r"^[0-9a-f]{40}$")
        for domain in ("metromap", "travelmap"):
            self.assertIn("_training_set.json", prepare_data.SOURCE_FILES[domain]["train"])
            self.assertIn("_test_set.json", prepare_data.SOURCE_FILES[domain]["test"])

    def test_locked_ids_map_to_expected_source_rows(self):
        for domain in ("metromap", "travelmap"):
            ids = prepare_data.locked_sample_ids(
                domain,
                include_prompt_examples=False,
            )
            self.assertEqual(len(ids), 2100)
            train_ids = [value for value in ids if ":train:" in value]
            test_ids = [value for value in ids if ":test:" in value]
            self.assertEqual(len(train_ids), 1700)
            self.assertEqual(len(test_ids), 400)
            for sample_id in ids:
                split, index = prepare_data.parse_sample_id(sample_id, domain)
                self.assertIn(split, {"train", "test"})
                self.assertGreaterEqual(index, 0)

    def test_asset_mapping_uses_official_and_runtime_layouts(self):
        rows = {
            "train": [
                {
                    "figure": "metromap/images/example.png",
                    "vertex_tab": "metromap/tabulars/example_vertex.json",
                }
            ],
            "test": [],
        }
        assets = prepare_data.required_assets_for_ids(
            "metromap",
            rows,
            ["metromap:train:000000"],
        )
        self.assertEqual(
            assets,
            {
                "assets/metromap/images/example.png": Path(
                    "metromap/images/example.png"
                ),
                "assets/metromap/tabulars/example_vertex.json": Path(
                    "metromap/tabulars/example_vertex.json"
                ),
            },
        )

    def test_smoke_selection_covers_train_and_test(self):
        for domain in ("metromap", "travelmap"):
            selected = prepare_data._smoke_sample_ids(domain, 2)
            self.assertEqual(len(selected), 4)
            self.assertEqual(
                sum(value.startswith(f"{domain}:train:") for value in selected),
                2,
            )
            self.assertEqual(
                sum(value.startswith(f"{domain}:test:") for value in selected),
                2,
            )

    def test_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            prepare_data._normalize_asset_reference(
                "metromap",
                "metromap/images/../../secret",
            )


if __name__ == "__main__":
    unittest.main()
