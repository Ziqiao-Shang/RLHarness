"""Load MapTab planning train/test (only_vertex2 by default)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rlharness.common.config import maptab_root

PLANNING_NAME = (
    "{domain}_shortest_path_query_map_and_tab_with_constraint_1_2_3_4"
    "{vertex_tag}_{split}_set.json"
)
COMBINED_ONLY_VERTEX2_NAME = (
    "{domain}_shortest_path_query_map_and_tab_with_constraint_1_2_3_4"
    "_only_vertex2.json"
)


def planning_json_path(domain: str, split: str, *, only_vertex2: bool = True) -> Path:
    root = maptab_root()
    split_dir = "training_set" if split == "train" else "test_set"
    split_name = "training" if split == "train" else "test"
    vertex_tag = "_only_vertex2" if only_vertex2 else ""
    name = PLANNING_NAME.format(domain=domain, vertex_tag=vertex_tag, split=split_name)
    path = root / domain / "data" / split_dir / name
    if path.exists():
        return path

    # Some domains ship only_vertex2 as one combined file with set_category.
    if only_vertex2:
        combined = (
            root
            / domain
            / "data"
            / "all"
            / COMBINED_ONLY_VERTEX2_NAME.format(domain=domain)
        )
        if combined.exists():
            return combined

        # Legacy fallback for datasets that have no only_vertex2 release.
        return planning_json_path(domain, split, only_vertex2=False)
    raise FileNotFoundError(path)


def load_planning(domain: str, split: str, *, only_vertex2: bool = True) -> list[dict[str, Any]]:
    path = planning_json_path(domain, split, only_vertex2=only_vertex2)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if only_vertex2 and path.parent.name == "all":
        category = "training_set" if split == "train" else "test_set"
        rows = [row for row in rows if row.get("set_category") == category]
    out = []
    for i, r in enumerate(rows):
        rec = dict(r)
        rec["domain"] = domain
        rec["split"] = split
        rec["sample_id"] = f"{domain}:{split}:{i:06d}"
        rec["gt_route"] = (rec.get("routes") or [""])[0]
        out.append(rec)
    return out
