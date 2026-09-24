#!/usr/bin/env python3
"""Validate and merge evaluation shards in source-data order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlharness.eval.eval_local import aggregate, load_input_rows, load_rows, select_sample_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("all", "metromap", "travelmap"), default="all")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--sample-ids-from", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    if args.input_json is not None and args.sample_ids_from is not None:
        raise SystemExit("Use either --input-json or --sample-ids-from, not both.")

    all_expected_rows = (
        load_input_rows(args.input_json, args.domain)
        if args.input_json
        else load_rows(args.domain, args.split)
    )
    if args.sample_ids_from is not None:
        all_expected_rows = select_sample_ids(all_expected_rows, args.sample_ids_from)
    if args.start < 0:
        raise SystemExit("--start must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    stop = None if args.limit is None else args.start + args.limit
    expected_rows = all_expected_rows[args.start:stop]
    expected_ids = [str(row["sample_id"]) for row in expected_rows]
    expected_set = set(expected_ids)
    records: dict[str, dict] = {}
    for path in args.inputs:
        if not path.is_file():
            raise SystemExit(f"Missing shard: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id") or "")
                if sample_id not in expected_set:
                    raise SystemExit(f"Unexpected sample_id at {path}:{line_number}: {sample_id}")
                if sample_id in records and records[sample_id] != record:
                    raise SystemExit(f"Conflicting duplicate sample_id: {sample_id}")
                records[sample_id] = record

    missing = [sample_id for sample_id in expected_ids if sample_id not in records]
    if missing:
        raise SystemExit(
            f"Only found {len(records)}/{len(expected_ids)} records; missing={missing[:5]}"
        )
    ordered = [records[sample_id] for sample_id in expected_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)

    summary = aggregate(ordered)
    first = ordered[0]
    summary.update(
        {
            "domain": args.domain,
            "split": args.split,
            "input_json": str(args.input_json) if args.input_json else None,
            "sample_ids_from": str(args.sample_ids_from) if args.sample_ids_from else None,
            "thinking": bool(first.get("thinking_enabled")),
            "reasoning_effort": first.get("reasoning_effort"),
            "model": first.get("model"),
            "adapter": first.get("adapter"),
            "predictions": str(args.output),
            "shards": [str(path) for path in args.inputs],
        }
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Merged {len(ordered)} records: {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
