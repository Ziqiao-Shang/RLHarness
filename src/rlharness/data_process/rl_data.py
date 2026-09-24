#!/usr/bin/env python3
"""Build canonical MapTab GRPO data from a fixed filtered subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


from rlharness.common.config import ROOT
from rlharness.common.prompt_build import build_prompt_bundle
from rlharness.data_process.rl_format import build_verl_row


EXCLUDED_DIFFICULTIES = {
    ("Medium", "Hard"),
    ("Hard", "Medium"),
    ("Hard", "Hard"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=("metromap", "travelmap"),
        default="metromap",
    )
    parser.add_argument("--subset-dir", type=Path)
    parser.add_argument("--prompt-template", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--table-max-chars", type=int, default=20_000)
    parser.add_argument("--image-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--image-min-pixels", type=int, default=3_136)
    args = parser.parse_args()
    if args.domain == "metromap":
        args.subset_dir = args.subset_dir or ROOT / "data/generated/splits"
        args.output_dir = args.output_dir or ROOT / "data/generated/rl_round1"
    else:
        args.subset_dir = args.subset_dir or ROOT / "data/generated/travelmap/splits"
        args.output_dir = args.output_dir or ROOT / "data/generated/travelmap/rl_round1"
    args.prompt_template = (
        args.prompt_template
        or ROOT / "prompts/student" / args.domain / "original.txt"
    )
    return args


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def difficulty_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("Map_Difficulty") or "Unknown"),
        str(row.get("Query_Difficulty") or "Unknown"),
    )


def starts_with_transfer(route: str) -> bool:
    first = str(route).split("-", 1)[0].strip().casefold()
    return first.endswith("(transfer)")


def prepare_rows(
    source: list[dict[str, Any]],
    *,
    domain: str,
    prompt_template: str,
    table_max_chars: int,
    image_max_pixels: int,
    image_min_pixels: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, record in enumerate(source):
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"Missing sample_id at source row {index}")
        record_domain = str(record.get("domain") or "")
        if record_domain != domain or not sample_id.startswith(f"{domain}:"):
            raise ValueError(
                f"Source row {sample_id!r} belongs to {record_domain!r}, "
                f"expected domain {domain!r}"
            )
        if difficulty_key(record) in EXCLUDED_DIFFICULTIES:
            raise ValueError(f"Excluded difficulty leaked into subset: {sample_id}")
        if domain == "metromap" and starts_with_transfer(
            str(record.get("gt_route") or "")
        ):
            raise ValueError(f"Start-transfer route leaked into subset: {sample_id}")

        bundle = build_prompt_bundle(
            record,
            include_skills=False,
            thinking=False,
            table_max_chars=table_max_chars,
            prompt_template=prompt_template,
        )
        row = build_verl_row(
            {
                **record,
                "index": index,
                "prompt": bundle["text"],
                "images": [bundle["image_path"]] if bundle["image_path"] else [],
                "table_path": bundle["table_path"],
            },
            image_max_pixels=image_max_pixels,
            image_min_pixels=image_min_pixels,
        )
        row["extra_info"].update(
            {
                "map_difficulty": difficulty_key(record)[0],
                "query_difficulty": difficulty_key(record)[1],
                "prompt_transform_version": f"{domain}_skill_reasoning_v1",
            }
        )
        output.append(row)
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def labeled_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(difficulty_key(row) for row in rows)
    return {
        f"{map_level}x{query_level}": count
        for (map_level, query_level), count in sorted(counts.items())
    }


def main() -> None:
    args = parse_args()
    if args.table_max_chars <= 0:
        raise SystemExit("--table-max-chars must be positive")
    prompt_template = args.prompt_template.read_text(encoding="utf-8").strip()

    train_source = read_json(args.subset_dir / "train_source.json")
    validation_source = read_json(args.subset_dir / "validation.json")
    if not isinstance(train_source, list) or not isinstance(validation_source, list):
        raise SystemExit("Subset sources must be JSON lists")

    train_ids = {str(row.get("sample_id") or "") for row in train_source}
    validation_ids = {str(row.get("sample_id") or "") for row in validation_source}
    if len(train_ids) != len(train_source):
        raise SystemExit("Duplicate sample IDs in training subset")
    if len(validation_ids) != len(validation_source):
        raise SystemExit("Duplicate sample IDs in validation subset")
    overlap = train_ids & validation_ids
    if overlap:
        raise SystemExit(f"Train/validation overlap: {sorted(overlap)[:5]}")

    common = {
        "domain": args.domain,
        "prompt_template": prompt_template,
        "table_max_chars": args.table_max_chars,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
    }
    train_rows = prepare_rows(train_source, **common)
    validation_rows = prepare_rows(validation_source, **common)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    manifest = {
        "domain": args.domain,
        "policy": "fixed_filtered_1600_train_100_validation",
        "native_thinking": False,
        "visible_reasoning": True,
        "output_contract": "<reasoning>...</reasoning><response>...</response>",
        "prompt_template": str(args.prompt_template.resolve()),
        "prompt_transform_version": f"{args.domain}_skill_reasoning_v1",
        "table_max_chars": args.table_max_chars,
        "image_max_pixels": args.image_max_pixels,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_validation_overlap": 0,
        "train_difficulty_counts": labeled_counts(train_source),
        "validation_difficulty_counts": labeled_counts(validation_source),
        "train_file": str(train_path.resolve()),
        "validation_file": str(validation_path.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
