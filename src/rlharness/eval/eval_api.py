#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible vision model on a fixed MapTab subset."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


from rlharness.common.api_client import multimodal_messages, openlux_client
from rlharness.common.data_load import load_planning
from rlharness.common.prompt_build import build_prompt_bundle
from rlharness.common.response_parse import extract_reasoning, extract_response, extract_think
from rlharness.common.route_score import score_route


class ThreadClients:
    """Create one API client per worker thread."""

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout
        self.local = threading.local()

    def get(self) -> Any:
        client = getattr(self.local, "client", None)
        if client is None:
            client = openlux_client(timeout=self.timeout)
            self.local.client = client
        return client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("metromap", "travelmap"), required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--sample-ids-from", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--table-max-chars", type=int, default=20_000)
    parser.add_argument("--image-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.retries <= 0:
        raise SystemExit("--retries must be positive")
    if args.retry_base_seconds < 0:
        raise SystemExit("--retry-base-seconds cannot be negative")
    if args.table_max_chars <= 0:
        raise SystemExit("--table-max-chars must be positive")
    if args.image_max_pixels <= 0:
        raise SystemExit("--image-max-pixels must be positive")
    if not args.sample_ids_from.is_file():
        raise SystemExit(f"Missing sample-ID manifest: {args.sample_ids_from}")
    if not args.prompt_template.is_file():
        raise SystemExit(f"Missing prompt template: {args.prompt_template}")


def read_sample_ids(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sample_ids = [
            str(value.get("sample_id") if isinstance(value, dict) else value)
            for value in values
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("sample_ids", []) if isinstance(payload, dict) else payload
        sample_ids = [str(value) for value in values]
    if not sample_ids or any(not value for value in sample_ids):
        raise SystemExit(f"No valid sample IDs in {path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit(f"Duplicate sample IDs in {path}")
    return sample_ids


def select_rows(domain: str, split: str, sample_ids: list[str]) -> list[dict[str, Any]]:
    rows = load_planning(domain, split)
    by_id = {str(row["sample_id"]): row for row in rows}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} sample IDs are absent from {domain}/{split}; examples={missing[:5]}"
        )
    return [by_id[sample_id] for sample_id in sample_ids]


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise SystemExit(f"Missing sample_id at {path}:{line_number}")
            rows[sample_id] = row
    return rows


def prepare_image(source: str, cache: Path, max_pixels: int) -> str:
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing map image: {path}")
    target = cache / f"{path.parent.name}_{path.stem}_{max_pixels}.png"
    if target.is_file():
        return str(target)

    cache.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        pixels = image.width * image.height
        if pixels > max_pixels:
            factor = math.sqrt(max_pixels / pixels)
            image = image.resize(
                (max(1, int(image.width * factor)), max(1, int(image.height * factor))),
                Image.Resampling.LANCZOS,
            )
        temporary = target.with_name(
            f".{target.name}.{threading.get_ident()}.tmp"
        )
        image.save(temporary, format="PNG")
        temporary.replace(target)
    return str(target)


def score_prediction(prediction: str, gt_route: str) -> dict[str, Any]:
    predicted_route = extract_response(prediction) or ""
    reasoning = extract_reasoning(prediction)
    thought = extract_think(prediction)
    route_scores = score_route(predicted_route, gt_route)
    format_ok = bool(predicted_route) and reasoning is not None and thought is None
    return {
        "reasoning": reasoning or "",
        "pred_route": predicted_route,
        "format_ok": format_ok,
        **route_scores,
    }


def usage_value(response: Any, name: str) -> int:
    usage = getattr(response, "usage", None)
    value = getattr(usage, name, 0) if usage is not None else 0
    return int(value or 0)


def evaluate_one(
    row: dict[str, Any],
    *,
    prompt_template: str,
    image_cache: Path,
    clients: ThreadClients,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_base_seconds: float,
    table_max_chars: int,
    image_max_pixels: int,
    seed: int,
    enable_thinking: bool | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    started = time.monotonic()
    bundle = build_prompt_bundle(
        row,
        thinking=False,
        table_max_chars=table_max_chars,
        prompt_template=prompt_template,
    )
    image_path = ""
    if bundle["image_path"]:
        image_path = prepare_image(
            str(bundle["image_path"]), image_cache, image_max_pixels
        )
    messages = multimodal_messages(
        system="",
        user_text=str(bundle["text"]),
        image_path=image_path or None,
    )
    attempts: list[dict[str, Any]] = []
    sample_id = str(row["sample_id"])

    for attempt in range(1, retries + 1):
        attempt_started = time.monotonic()
        try:
            response = clients.get().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **({"extra_body": {"enable_thinking": enable_thinking}} if enable_thinking is not None else {}),
            )
            choice = response.choices[0]
            prediction = (choice.message.content or "").strip()
            scored = score_prediction(prediction, str(row["gt_route"]))
            record = {
                "sample_id": sample_id,
                "domain": str(row["domain"]),
                "split": str(row.get("split") or row.get("set_category") or ""),
                "question": str(row.get("question") or ""),
                "Map_Difficulty": str(row.get("Map_Difficulty") or "Unknown"),
                "Query_Difficulty": str(row.get("Query_Difficulty") or "Unknown"),
                "gt_route": str(row["gt_route"]),
                "prediction": prediction,
                **scored,
                "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
                "prompt_tokens": usage_value(response, "prompt_tokens"),
                "response_tokens": usage_value(response, "completion_tokens"),
                "seconds": round(time.monotonic() - started, 3),
                "attempt_count": attempt,
                "model": model,
                "enable_thinking": enable_thinking,
                "backend": "openai_compatible_api",
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            attempts.append(
                {
                    "sample_id": sample_id,
                    "attempt": attempt,
                    "ok": True,
                    "seconds": round(time.monotonic() - attempt_started, 3),
                }
            )
            return record, attempts
        except Exception as exc:
            attempts.append(
                {
                    "sample_id": sample_id,
                    "attempt": attempt,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.monotonic() - attempt_started, 3),
                }
            )
            if attempt < retries:
                jitter = random.Random(f"{seed}:{sample_id}:{attempt}").uniform(0.0, 1.0)
                delay = min(retry_base_seconds * (2 ** (attempt - 1)) + jitter, 60.0)
                time.sleep(delay)
    return None, attempts


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    denominator = max(count, 1)
    return {
        "count": count,
        "hard_correct": sum(
            bool(row.get("format_ok")) and float(row.get("all_acc", 0.0)) >= 1.0
            for row in rows
        ),
        "all_acc": sum(float(row.get("all_acc", 0.0)) for row in rows) / denominator,
        "part_acc": sum(float(row.get("part_acc", 0.0)) for row in rows) / denominator,
        "format_rate": sum(bool(row.get("format_ok")) for row in rows) / denominator,
        "truncated": sum(str(row.get("finish_reason")) == "length" for row in rows),
        "avg_prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in rows)
        / denominator,
        "avg_response_tokens": sum(int(row.get("response_tokens", 0)) for row in rows)
        / denominator,
    }


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    unresolved: int,
    started: float,
) -> None:
    ordered = [
        completed[str(row["sample_id"])]
        for row in selected
        if str(row["sample_id"]) in completed
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        key = f"{row['Map_Difficulty']}x{row['Query_Difficulty']}"
        groups[key].append(row)
    payload = {
        **metrics(ordered),
        "selected": len(selected),
        "unresolved": unresolved,
        "difficulty_breakdown": {
            name: metrics(rows) for name, rows in sorted(groups.items())
        },
        "model": args.model,
        "prompt_template": str(args.prompt_template.resolve()),
        "sample_ids_from": str(args.sample_ids_from.resolve()),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
        "enable_thinking": args.enable_thinking,
        "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.monotonic()
    sample_ids = read_sample_ids(args.sample_ids_from)
    selected = select_rows(args.domain, args.split, sample_ids)
    selected_ids = set(sample_ids)
    prompt_template = args.prompt_template.read_text(encoding="utf-8").strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing(args.output) if args.resume else {}
    completed = {
        sample_id: row
        for sample_id, row in existing.items()
        if sample_id in selected_ids
    }
    pending = [row for row in selected if str(row["sample_id"]) not in completed]
    mode = "a" if args.resume and args.output.is_file() else "w"
    summary_path = args.output.with_suffix(".summary.json")
    attempts_path = args.output.with_suffix(".attempts.jsonl")
    print(
        f"OpenLux rows={len(selected)} complete={len(completed)} pending={len(pending)} "
        f"model={args.model} workers={args.workers}",
        flush=True,
    )

    if not pending:
        write_summary(
            summary_path,
            args=args,
            selected=selected,
            completed=completed,
            unresolved=0,
            started=started,
        )
        print(f"Nothing to resume. Summary: {summary_path}")
        return

    clients = ThreadClients(timeout=args.timeout)
    unresolved = 0
    image_cache = args.output.parent / "image_cache"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(mode, encoding="utf-8") as output_handle, attempts_path.open(
        "a", encoding="utf-8"
    ) as attempts_handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                row,
                prompt_template=prompt_template,
                image_cache=image_cache,
                clients=clients,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=args.retries,
                retry_base_seconds=args.retry_base_seconds,
                table_max_chars=args.table_max_chars,
                image_max_pixels=args.image_max_pixels,
                seed=args.seed,
                enable_thinking=args.enable_thinking,
            ): str(row["sample_id"])
            for row in pending
        }
        finished = 0
        for future in as_completed(futures):
            sample_id = futures[future]
            record, attempts = future.result()
            for attempt in attempts:
                attempts_handle.write(json.dumps(attempt, ensure_ascii=False) + "\n")
            attempts_handle.flush()
            if record is None:
                unresolved += 1
                last_error = attempts[-1].get("error", "unknown error")
                print(f"[{finished + 1}/{len(pending)}] {sample_id} FAIL {last_error}", flush=True)
            else:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                completed[sample_id] = record
                hard = int(bool(record["format_ok"]) and float(record["all_acc"]) >= 1.0)
                print(f"[{finished + 1}/{len(pending)}] {sample_id} hard={hard}", flush=True)
            finished += 1

    write_summary(
        summary_path,
        args=args,
        selected=selected,
        completed=completed,
        unresolved=unresolved,
        started=started,
    )
    if unresolved:
        raise SystemExit(
            f"{unresolved} API evaluations remain unresolved; rerun the same command to resume"
        )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
