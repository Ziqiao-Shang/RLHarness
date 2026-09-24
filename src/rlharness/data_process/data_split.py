#!/usr/bin/env python3
"""Create a canonical filtered Stage 2 subset used by later experiments."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rlharness.data_process.sft_teacher import (
    gt_starts_with_transfer,
    select_difficulty_quotas,
)
from rlharness.common.data_load import load_planning


EXCLUDED_DIFFICULTIES = {
    ("Medium", "Hard"),
    ("Hard", "Medium"),
    ("Hard", "Hard"),
}

# These TravelMap rows have malformed destination aliases and aggregate prices
# that do not match the published route, so they are unsafe supervision labels.
KNOWN_BAD_SAMPLE_IDS = {
    "travelmap": {
        "travelmap:train:000636",
        "travelmap:train:000645",
    },
}

# These samples are embedded as TravelMap few-shot demonstrations. Keep the
# exact rows out of every split, and reserve their figures for the training
# side so validation/test cannot reuse demonstration topology.
PROMPT_DEMONSTRATION_SAMPLE_IDS = {
    "travelmap": {
        "travelmap:train:003184",
        "travelmap:train:005156",
    },
}


def difficulty_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("Map_Difficulty") or "Medium"),
        str(row.get("Query_Difficulty") or "Medium"),
    )


def proportional_quotas(
    rows: list[dict[str, Any]], target: int
) -> dict[tuple[str, str], int]:
    counts = Counter(map(difficulty_key, rows))
    total = sum(counts.values())
    exact = {key: target * count / total for key, count in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - quotas[key], counts[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def choose_validation_figure_pool(
    rows: list[dict[str, Any]],
    quotas: dict[tuple[str, str], int],
    *,
    seed: int,
    forbidden_figures: set[str],
    min_figures_per_map_difficulty: int = 6,
) -> set[str]:
    """Choose whole figures that can satisfy every validation stratum."""
    by_figure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        figure = str(row.get("figure") or "")
        if figure and figure not in forbidden_figures:
            by_figure[figure].append(row)

    rng = random.Random(seed)
    figures = sorted(by_figure)
    rng.shuffle(figures)
    tie_rank = {figure: index for index, figure in enumerate(figures)}
    available: Counter[tuple[str, str]] = Counter()
    selected: set[str] = set()

    while any(available[key] < quota for key, quota in quotas.items()):
        deficits = {
            key: max(0, quota - available[key]) for key, quota in quotas.items()
        }
        best_figure = None
        best_gain = 0
        for figure in figures:
            if figure in selected:
                continue
            counts = Counter(difficulty_key(row) for row in by_figure[figure])
            gain = sum(min(counts[key], deficits[key]) for key in quotas)
            if gain > best_gain or (
                gain == best_gain
                and gain > 0
                and best_figure is not None
                and tie_rank[figure] < tie_rank[best_figure]
            ):
                best_figure = figure
                best_gain = gain
        if best_figure is None:
            missing = {
                f"{key[0]}x{key[1]}": quota - available[key]
                for key, quota in quotas.items()
                if available[key] < quota
            }
            raise RuntimeError(
                f"Cannot satisfy validation quotas with disjoint figures: {missing}"
            )
        selected.add(best_figure)
        available.update(difficulty_key(row) for row in by_figure[best_figure])

    figure_map_difficulty = {
        figure: difficulty_key(figure_rows[0])[0]
        for figure, figure_rows in by_figure.items()
    }
    for map_difficulty in sorted({key[0] for key in quotas}):
        current = sum(
            figure_map_difficulty[figure] == map_difficulty
            for figure in selected
        )
        candidates = [
            figure
            for figure in figures
            if figure not in selected
            and figure_map_difficulty[figure] == map_difficulty
        ]
        needed = max(0, min_figures_per_map_difficulty - current)
        if len(candidates) < needed:
            raise RuntimeError(
                f"Need {needed} more {map_difficulty} validation figures, "
                f"but only {len(candidates)} remain"
            )
        selected.update(candidates[:needed])

    return selected


def eligible(domain: str, split: str) -> list[dict[str, Any]]:
    bad_ids = KNOWN_BAD_SAMPLE_IDS.get(domain, set())
    return [
        row
        for row in load_planning(domain, split, only_vertex2=True)
        if difficulty_key(row) not in EXCLUDED_DIFFICULTIES
        and (domain != "metromap" or not gt_starts_with_transfer(row))
        and str(row["sample_id"]) not in bad_ids
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def materialize_fixed(domain: str, source: Path, output: Path) -> dict:
    """Rebuild data rows from locked IDs without resampling historical splits."""
    names = ("train", "validation", "test")
    ids = {}
    for name, count in zip(names, (1600, 100, 400)):
        values = json.loads((source / f"{name}_sample_ids.json").read_text())
        if (not isinstance(values, list) or len(values) != count
                or any(not isinstance(v, str) for v in values) or len(set(values)) != count):
            raise ValueError(f"Expected {count} unique locked {name} IDs")
        ids[name] = values
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(ids[a]) & set(ids[b]):
            raise ValueError(f"Locked {a}/{b} IDs overlap")
    pools = {split: {r["sample_id"]: r for r in load_planning(domain, split)}
             for split in ("train", "test")}
    rows = {}
    for name in names:
        pool = pools["test" if name == "test" else "train"]
        if not set(ids[name]) <= pool.keys():
            raise ValueError(f"Unknown or wrong-domain/source {name} IDs")
        rows[name] = [pool[sid] for sid in ids[name]]
        old = output / f"{name}_sample_ids.json"
        if old.exists() and json.loads(old.read_text()) != ids[name]:
            raise ValueError("Output has different split IDs; use a new output directory")
    figures = {name: {r["figure"] for r in values} for name, values in rows.items()}
    manifest = {"domain": domain, "policy": "reuse_locked_ids_no_resampling",
                "source_ids_dir": str(source.resolve()),
                "train_rows": 1600, "validation_rows": 100, "test_rows": 400,
                "figure_overlap": {f"{a}_{b}": len(figures[a] & figures[b])
                                   for a, b in (("train", "validation"), ("train", "test"),
                                                ("validation", "test"))}}
    for name in names:
        write_json(output / f"{name}_sample_ids.json", ids[name])
        write_json(output / ("train_source.json" if name == "train" else f"{name}.json"), rows[name])
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=("metromap", "travelmap"),
        default="metromap",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--fixed-ids-dir", type=Path, help="Reuse locked Train1600/Val100/Test400 IDs")
    parser.add_argument(
        "--allow-resample",
        action="store_true",
        help="Development only: create a new split from the full source pools.",
    )
    args = parser.parse_args()

    if args.fixed_ids_dir:
        if args.allow_resample:
            parser.error("--fixed-ids-dir and --allow-resample are mutually exclusive")
        print(json.dumps(materialize_fixed(args.domain, args.fixed_ids_dir, args.output_dir),
                         ensure_ascii=False, indent=2))
        return

    if not args.allow_resample:
        parser.error(
            "Refusing to resample the experiment data. Pass --fixed-ids-dir with the "
            "locked Train1600/Val100/Test400 IDs. --allow-resample is for development only."
        )

    raw_train_pool = eligible(args.domain, "train")
    test_pool = eligible(args.domain, "test")

    demonstration_ids = PROMPT_DEMONSTRATION_SAMPLE_IDS.get(args.domain, set())
    demonstration_rows = [
        row for row in raw_train_pool if str(row["sample_id"]) in demonstration_ids
    ]
    found_demonstration_ids = {
        str(row["sample_id"]) for row in demonstration_rows
    }
    missing_demonstration_ids = demonstration_ids - found_demonstration_ids
    if missing_demonstration_ids:
        raise RuntimeError(
            "Prompt demonstration samples are missing from the source pool: "
            f"{sorted(missing_demonstration_ids)}"
        )
    reserved_train_figures = {
        str(row["figure"]) for row in demonstration_rows
    }
    train_source_pool = [
        row
        for row in raw_train_pool
        if str(row["sample_id"]) not in demonstration_ids
    ]

    validation_quotas = proportional_quotas(
        train_source_pool, args.validation_size
    )
    validation_figure_pool = choose_validation_figure_pool(
        train_source_pool,
        validation_quotas,
        seed=args.seed + 1,
        forbidden_figures=reserved_train_figures,
    )
    validation_candidates = [
        row
        for row in train_source_pool
        if str(row["figure"]) in validation_figure_pool
    ]
    validation_rows = select_difficulty_quotas(
        validation_candidates,
        quotas=validation_quotas,
        seed=args.seed + 2,
    )
    validation_figures = {str(row["figure"]) for row in validation_rows}

    train_pool = [
        row
        for row in train_source_pool
        if str(row["figure"]) not in validation_figures
    ]
    train_quotas = proportional_quotas(train_source_pool, args.train_size)
    train_rows = select_difficulty_quotas(
        train_pool,
        quotas=train_quotas,
        seed=args.seed,
    )

    test_quotas = proportional_quotas(test_pool, args.test_size)
    test_rows = select_difficulty_quotas(
        test_pool,
        quotas=test_quotas,
        seed=args.seed + 2,
    )

    split_rows = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }
    figure_sets = {
        name: {str(row["figure"]) for row in rows}
        for name, rows in split_rows.items()
    }
    figure_overlaps = {
        "train_validation": figure_sets["train"] & figure_sets["validation"],
        "train_test": figure_sets["train"] & figure_sets["test"],
        "validation_test": figure_sets["validation"] & figure_sets["test"],
    }
    if any(figure_overlaps.values()):
        raise RuntimeError(
            "Figure leakage detected: "
            + repr({key: sorted(value) for key, value in figure_overlaps.items()})
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "train_source.json", train_rows)
    write_json(args.output_dir / "validation.json", validation_rows)
    write_json(args.output_dir / "test.json", test_rows)
    write_json(
        args.output_dir / "train_sample_ids.json",
        [row["sample_id"] for row in train_rows],
    )
    write_json(
        args.output_dir / "validation_sample_ids.json",
        [row["sample_id"] for row in validation_rows],
    )
    write_json(
        args.output_dir / "test_sample_ids.json",
        [row["sample_id"] for row in test_rows],
    )

    def labeled(quotas: dict[tuple[str, str], int]) -> dict[str, int]:
        return {
            f"{map_level}x{query_level}": count
            for (map_level, query_level), count in sorted(quotas.items())
        }

    manifest = {
        "domain": args.domain,
        "policy": "exclude_medium_hard_hard_medium_hard_hard",
        "excluded_difficulties": [
            f"{a}x{b}" for a, b in sorted(EXCLUDED_DIFFICULTIES)
        ],
        "exclude_start_transfer": args.domain == "metromap",
        "excluded_known_bad_sample_ids": sorted(
            KNOWN_BAD_SAMPLE_IDS.get(args.domain, set())
        ),
        "seed": args.seed,
        "eligible_train_rows": len(raw_train_pool),
        "eligible_test_rows": len(test_pool),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "train_quotas": labeled(train_quotas),
        "validation_quotas": labeled(validation_quotas),
        "test_quotas": labeled(test_quotas),
        "split_unit": "figure",
        "prompt_demonstration_sample_ids": sorted(demonstration_ids),
        "reserved_train_figures": sorted(reserved_train_figures),
        "train_figures": len(figure_sets["train"]),
        "validation_figures": len(figure_sets["validation"]),
        "test_figures": len(figure_sets["test"]),
        "figure_overlap": {
            key: len(value) for key, value in figure_overlaps.items()
        },
        "prompt_demonstration_rows_in_splits": sum(
            str(row["sample_id"]) in demonstration_ids
            for rows in split_rows.values()
            for row in rows
        ),
        "prompt_demonstration_figures_in_validation_or_test": len(
            reserved_train_figures
            & (figure_sets["validation"] | figure_sets["test"])
        ),
        "train_ids": len({str(row["sample_id"]) for row in train_rows}),
        "train_validation_overlap": len(
            {str(row["sample_id"]) for row in train_rows}
            & {str(row["sample_id"]) for row in validation_rows}
        ),
        "train_test_overlap": len(
            {str(row["sample_id"]) for row in train_rows}
            & {str(row["sample_id"]) for row in test_rows}
        ),
        "validation_test_overlap": len(
            {str(row["sample_id"]) for row in validation_rows}
            & {str(row["sample_id"]) for row in test_rows}
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
