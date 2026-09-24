#!/usr/bin/env python3
"""Re-render audited Stage 2 labels with the current student prompt."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rlharness.common.prompt_build import build_prompt_bundle
from rlharness.data_process.sft_teacher import load_token_counter


SAMPLE_ID_RE = re.compile(r"^(metromap|travelmap):train:(\d{6})$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def sample_index(sample_id: str, raw_size: int, domain: str) -> int:
    match = SAMPLE_ID_RE.fullmatch(sample_id)
    if not match:
        raise ValueError(f"Unsupported sample ID: {sample_id}")
    if match.group(1) != domain:
        raise ValueError(f"Expected {domain} sample ID, got: {sample_id}")
    index = int(match.group(2))
    if index >= raw_size:
        raise ValueError(f"Sample index is out of range: {sample_id}")
    return index


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("metromap", "travelmap"), default="metromap")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--raw-training-data", type=Path, required=True)
    parser.add_argument("--fixed-train-ids", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table-max-chars", type=int, default=20000)
    parser.add_argument("--cutoff-len", type=int, default=20480)
    args = parser.parse_args()

    labels = read_jsonl(args.labels)
    raw_rows = read_json(args.raw_training_data)
    fixed_train_ids = set(read_json(args.fixed_train_ids))
    validation_ids = set(read_json(args.validation_ids))
    test_ids = set(read_json(args.test_ids))
    validation_rows = read_json(args.validation_file)
    prompt_template = args.prompt_template.read_text(encoding="utf-8").strip()
    tokenizer = load_token_counter(args.tokenizer)

    if not isinstance(raw_rows, list) or not isinstance(validation_rows, list):
        raise ValueError("Raw training and validation data must be JSON arrays")
    raw_rows_by_id = {
        str(row["sample_id"]): row
        for row in raw_rows
        if isinstance(row, dict) and row.get("sample_id")
    }

    rendered: list[dict[str, Any]] = []
    seen: set[str] = set()
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    total_text_lengths: list[int] = []
    difficulty = Counter()

    for label in labels:
        sample_id = str(label.get("sample_id") or "")
        if sample_id in seen:
            raise ValueError(f"Duplicate label: {sample_id}")
        if sample_id not in fixed_train_ids:
            raise ValueError(f"Label is outside the fixed training subset: {sample_id}")
        if sample_id in validation_ids or sample_id in test_ids:
            raise ValueError(f"Evaluation leakage detected: {sample_id}")
        seen.add(sample_id)

        if raw_rows_by_id:
            if sample_id not in raw_rows_by_id:
                raise ValueError(f"Label is absent from prepared training source: {sample_id}")
            raw = dict(raw_rows_by_id[sample_id])
        else:
            raw = dict(raw_rows[sample_index(sample_id, len(raw_rows), args.domain)])
        raw.update(
            {
                "domain": args.domain,
                "split": "train",
                "sample_id": sample_id,
                "gt_route": str(label["gt_route"]),
            }
        )
        bundle = build_prompt_bundle(
            raw,
            include_skills=False,
            thinking=False,
            table_max_chars=args.table_max_chars,
            prompt_template=prompt_template,
        )
        image_path = bundle["image_path"]
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError(f"Missing image for {sample_id}: {image_path}")

        output = str(label.get("output") or "").strip()
        if not (output.startswith("<reasoning>") and output.endswith("</response>")):
            raise ValueError(f"Invalid teacher output contract: {sample_id}")
        if str(label["gt_route"]) not in output:
            raise ValueError(f"Teacher output does not contain its GT route: {sample_id}")

        human = "<image>\n" + bundle["text"]
        updated = dict(label)
        updated["student_prompt"] = bundle["text"]
        updated["image_path"] = image_path
        updated["conversations"] = [
            {"from": "human", "value": human},
            {"from": "gpt", "value": output},
        ]
        updated["images"] = [image_path]
        updated["student_prompt_template"] = str(args.prompt_template.resolve())
        rendered.append(updated)

        prompt_tokens = len(tokenizer.encode(bundle["text"], add_special_tokens=False))
        response_tokens = len(tokenizer.encode(output, add_special_tokens=False))
        prompt_lengths.append(prompt_tokens)
        response_lengths.append(response_tokens)
        total_text_lengths.append(prompt_tokens + response_tokens)
        difficulty[
            f"{str(raw.get('Map_Difficulty')).title()}x"
            f"{str(raw.get('Query_Difficulty')).title()}"
        ] += 1

    rendered.sort(key=lambda row: str(row["sample_id"]))
    if len(rendered) != len(seen):
        raise AssertionError("Rendered rows are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "train.json", rendered)
    write_jsonl(args.output_dir / "train.jsonl", rendered)
    write_json(args.output_dir / "validation.raw.json", validation_rows)
    write_json(
        args.output_dir / "manifest.json",
        {
            "train_rows": len(rendered),
            "validation_rows": len(validation_rows),
            "sample_overlap": len(seen & validation_ids),
            "test_overlap": len(seen & test_ids),
            "prompt_template": str(args.prompt_template.resolve()),
            "source_labels": str(args.labels.resolve()),
            "table_max_chars": args.table_max_chars,
            "cutoff_len": args.cutoff_len,
            "difficulty_distribution": dict(sorted(difficulty.items())),
            "text_token_lengths_excluding_image_tokens": {
                "prompt_min": min(prompt_lengths),
                "prompt_median": int(statistics.median(prompt_lengths)),
                "prompt_p95": percentile(prompt_lengths, 0.95),
                "prompt_max": max(prompt_lengths),
                "response_min": min(response_lengths),
                "response_median": int(statistics.median(response_lengths)),
                "response_p95": percentile(response_lengths, 0.95),
                "response_max": max(response_lengths),
                "combined_min": min(total_text_lengths),
                "combined_median": int(statistics.median(total_text_lengths)),
                "combined_p95": percentile(total_text_lengths, 0.95),
                "combined_max": max(total_text_lengths),
                "combined_over_cutoff": sum(
                    length > args.cutoff_len for length in total_text_lengths
                ),
            },
        },
    )

    print(f"train={len(rendered)} validation={len(validation_rows)}")
    print(f"sample_overlap={len(seen & validation_ids)} test_overlap={len(seen & test_ids)}")
    print(
        "combined_text_tokens="
        f"median={int(statistics.median(total_text_lengths))} "
        f"p95={percentile(total_text_lengths, 0.95)} "
        f"max={max(total_text_lengths)} "
        f"over_cutoff={sum(length > args.cutoff_len for length in total_text_lengths)}"
    )
    print(f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
