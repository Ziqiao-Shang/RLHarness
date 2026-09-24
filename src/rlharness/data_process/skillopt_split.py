"""Sample SkillOpt optimization IDs from a locked training pool, without resplitting."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from rlharness.common.config import ROOT
from rlharness.common.data_load import load_planning
from rlharness.data_process.data_split import difficulty_key, proportional_quotas


def read_ids(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))):
        raise ValueError(f"Expected nonempty, unique sample IDs: {path}")
    return values


def select(rows: list[dict], count: int, seed: int) -> list[dict]:
    if not 0 < count <= len(rows):
        raise ValueError("Sample count must fit inside the locked training pool")
    groups = defaultdict(list)
    for row in rows:
        groups[difficulty_key(row)].append(row)
    quotas = proportional_quotas(rows, count)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        selected.extend(rng.sample(sorted(groups[key], key=lambda r: r["sample_id"]), quotas[key]))
    rng.shuffle(selected)
    return selected


def prepare(domain: str, split_dir: Path, output_dir: Path, *, count: int = 480,
            batch_size: int = 40, seed: int = 42) -> dict:
    if batch_size < 1 or count % batch_size:
        raise ValueError("Sample count must be divisible by the positive step batch size")
    names = ("train", "validation", "test")
    paths = {name: split_dir / f"{name}_sample_ids.json" for name in names}
    pools = {name: read_ids(path) for name, path in paths.items()}
    reference_dir = ROOT / "data" / "reference" / domain / "splits"
    reference = {
        name: read_ids(reference_dir / f"{name}_sample_ids.json")
        for name in names
    }
    if pools != reference:
        raise ValueError(
            "SkillOpt input must exactly match the published Train1600/Val100/Test400 IDs"
        )
    if len(pools["train"]) != 1600:
        raise ValueError("This protocol requires the existing Train1600")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(pools[a]) & set(pools[b]):
            raise ValueError(f"Locked {a}/{b} sample IDs overlap; no automatic resplitting")
    source_train = {row["sample_id"]: row for row in load_planning(domain, "train")}
    source = {**source_train, **{row["sample_id"]: row for row in load_planning(domain, "test")}}
    if not set(pools["train"]) <= source_train.keys():
        raise ValueError("Train1600 contains unknown or non-training-source IDs")
    if not set(pools["validation"] + pools["test"]) <= source.keys():
        raise ValueError("Unknown held-out sample IDs for this domain")
    chosen = select([source_train[key] for key in pools["train"]], count, seed)
    chosen_ids = [row["sample_id"] for row in chosen]
    figure_sets = {
        name: {source[key]["figure"] for key in values}
        for name, values in {**pools, "optimization": chosen_ids}.items()
    }
    distribution = Counter(difficulty_key(row) for row in chosen)
    manifest = {
        "domain": domain, "seed": seed, "source_train_count": 1600,
        "optimization_count": count, "validation_count": len(pools["validation"]),
        "test_count": len(pools["test"]), "batch_size": batch_size,
        "steps": count // batch_size, "epochs": 1,
        "policy": "difficulty_proportional_without_replacement_from_locked_train1600",
        "validation_unchanged": True, "test_unchanged": True,
        "source_files": {name: {"path": str(path.resolve())} for name, path in paths.items()},
        "difficulty_counts": {f"{a}x{b}": n for (a, b), n in sorted(distribution.items())},
        "sample_id_overlap": {"optimization_validation": 0, "optimization_test": 0},
        "figure_overlap": {f"{a}_{b}": len(figure_sets[a] & figure_sets[b])
                           for a, b in (("train", "validation"), ("train", "test"),
                                        ("optimization", "validation"), ("optimization", "test"))},
        "note": "Map overlap is reported, not silently changed. This command does not run SkillOpt.",
    }
    def encoded(value: object) -> bytes:
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    outputs = {
        "train_sample_ids.json": encoded(chosen_ids),
        "validation_sample_ids.json": paths["validation"].read_bytes(),
        "test_sample_ids.json": paths["test"].read_bytes(),
        "batches.json": encoded([chosen_ids[i:i + batch_size] for i in range(0, count, batch_size)]),
        "manifest.json": encoded(manifest),
    }
    # Refuse to overwrite another selection, including the original split directory.
    for name, content in outputs.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Existing output differs; use a new output directory: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (output_dir / name).write_bytes(content)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("metromap", "travelmap"))
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.domain, args.split_dir, args.output_dir,
                       count=args.count, batch_size=args.batch_size, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
