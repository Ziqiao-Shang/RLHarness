#!/usr/bin/env python3
"""Prepare and mine Qwen RL trajectories for Skill Bank candidate generation."""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


from rlharness.common.api_client import extract_json_obj, openlux_client
from rlharness.common.config import ROOT


DEFAULT_PROMPT = ROOT / "prompts/student/metromap/original.txt"
DEFAULT_OUTPUT = ROOT / "artifacts/skill_evolution/evidence128"
MINING_PROMPT = ROOT / "prompts/evolution/metromap/analyze.txt"


class ThreadClients:
    def __init__(self, timeout: float) -> None:
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
    parser.add_argument("command", choices=("prepare", "mine", "all"))
    parser.add_argument("--s0", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path, default=MINING_PROMPT)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-reasoning-chars", type=int, default=6500)
    parser.add_argument(
        "--target-reports",
        type=int,
        default=0,
        help="For mining, stop after this many total reports; zero means all prepared batches.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing JSONL: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_rollout_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map a standard evaluation row to the evidence-mining schema."""
    sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
    if not sample_id:
        raise ValueError("Rollout row is missing sample_id")
    map_difficulty = str(row.get("Map_Difficulty") or "Unknown")
    query_difficulty = str(row.get("Query_Difficulty") or "Unknown")
    return {
        "id": sample_id,
        "hard": int(bool(row.get("all_acc", row.get("hard", 0)))),
        "format_ok": bool(row.get("format_ok")),
        "part_acc": float(row.get("part_acc") or 0),
        "source_finish_reason": str(row.get("finish_reason") or "unknown"),
        "task_type": f"{map_difficulty}x{query_difficulty}",
        "task_description": str(row.get("question") or row.get("task_description") or ""),
        "ground_truth_route": str(row.get("gt_route") or row.get("ground_truth_route") or ""),
        "predicted_route": str(row.get("pred_route") or row.get("predicted_route") or ""),
        "response": str(row.get("prediction") or row.get("response") or ""),
    }


def load_analysis_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [canonical_rollout_row(row) for row in load_jsonl(args.rollouts)]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Analysis input contains duplicate sample IDs")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract_section(text: str, heading: str, next_heading: str) -> str:
    try:
        return text.split(f"# {heading}\n", 1)[1].split(f"\n# {next_heading}\n", 1)[0].strip()
    except IndexError as exc:
        raise ValueError(f"Cannot extract section {heading!r}") from exc


def fill_template(template: str, **values: str) -> str:
    """Replace named placeholders without interpreting JSON example braces."""
    rendered = template
    for name, value in values.items():
        marker = "{" + name + "}"
        if marker not in rendered:
            raise ValueError(f"Template is missing placeholder {marker}")
        rendered = rendered.replace(marker, value)
    return rendered


def compact_reasoning(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head].rstrip() + "\n...[middle omitted for batching]...\n" + text[-tail:].lstrip()


def trajectory_text(row: dict[str, Any], max_chars: int) -> str:
    case_id = str(row["id"])
    response = str(row.get("response") or "")
    if not response:
        raise ValueError(f"Trajectory {case_id} has no response text")
    return "\n".join(
        [
            f"## Trajectory {case_id}",
            f"Difficulty: {row.get('task_type', 'Unknown')}",
            f"Correct: {bool(row.get('hard'))}",
            f"Format valid: {bool(row.get('format_ok'))}",
            f"Part accuracy: {float(row.get('part_acc') or 0):.6f}",
            f"Finish reason: {row.get('source_finish_reason') or 'unknown'}",
            f"Question: {row.get('task_description') or ''}",
            f"Ground-truth route: {row.get('ground_truth_route') or ''}",
            f"Predicted route: {row.get('predicted_route') or '(not parsed)'}",
            "Student response:",
            compact_reasoning(response, max_chars),
        ]
    )


def stratified_select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ValueError("sample-count must be positive and no larger than the corpus")
    rng = random.Random(seed)
    target_success = count // 2
    targets = {1: target_success, 0: count - target_success}
    selected: list[dict[str, Any]] = []
    for hard, target in targets.items():
        subset = [row for row in rows if int(row.get("hard") or 0) == hard]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            groups[str(row.get("task_type") or "Unknown")].append(row)
        for values in groups.values():
            rng.shuffle(values)
        total = len(subset)
        quotas = {key: target * len(values) / total for key, values in groups.items()}
        allocations = {key: min(len(groups[key]), int(value)) for key, value in quotas.items()}
        remaining = target - sum(allocations.values())
        order = sorted(groups, key=lambda key: (quotas[key] - allocations[key], len(groups[key])), reverse=True)
        while remaining:
            progressed = False
            for key in order:
                if allocations[key] < len(groups[key]):
                    allocations[key] += 1
                    remaining -= 1
                    progressed = True
                    if not remaining:
                        break
            if not progressed:
                raise ValueError("Could not allocate stratified sample")
        for key in sorted(groups):
            selected.extend(groups[key][: allocations[key]])
    rng.shuffle(selected)
    return selected


def mixed_batches(rows: list[dict[str, Any]], batch_size: int, seed: int) -> list[list[dict[str, Any]]]:
    if batch_size < 2:
        raise ValueError("batch-size must be at least 2")
    rng = random.Random(seed)
    success = [row for row in rows if int(row.get("hard") or 0) == 1]
    failure = [row for row in rows if int(row.get("hard") or 0) == 0]
    rng.shuffle(success)
    rng.shuffle(failure)
    batches: list[list[dict[str, Any]]] = []
    half = batch_size // 2
    while success or failure:
        batch = success[:half] + failure[: batch_size - half]
        del success[:half]
        del failure[: batch_size - half]
        if len(batch) < batch_size:
            source = success if success else failure
            need = batch_size - len(batch)
            batch.extend(source[:need])
            del source[:need]
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def prepare(args: argparse.Namespace) -> None:
    rows = load_analysis_rows(args)
    selected = stratified_select(rows, args.sample_count, args.seed)
    batches = mixed_batches(selected, args.batch_size, args.seed + 1)
    selection = {
        "seed": args.seed,
        "source": str(args.rollouts.resolve()),
        "sample_count": len(selected),
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "correctness": dict(Counter("success" if row["hard"] else "failure" for row in selected)),
        "difficulty": dict(sorted(Counter(str(row.get("task_type")) for row in selected).items())),
        "sample_ids": [row["id"] for row in selected],
    }
    write_json(args.output / "selection.json", selection)
    batch_dir = args.output / "teacher_analysis/batches"
    for index, batch in enumerate(batches):
        write_json(batch_dir / f"batch_{index:03d}.input.json", batch)
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def valid_mining(value: Any, batch_ids: set[str]) -> bool:
    if not isinstance(value, dict):
        return False
    required = ("successful_strategies", "failure_patterns", "existing_skill_findings", "missing_strategies", "anti_patterns")
    if any(not isinstance(value.get(key), list) for key in required):
        return False
    for key in required[:-1]:
        for item in value[key]:
            if not isinstance(item, dict):
                return False
            evidence = item.get("evidence_ids", [])
            if not isinstance(evidence, list) or any(str(case_id) not in batch_ids for case_id in evidence):
                return False
            if len({str(case_id) for case_id in evidence}) < 2:
                return False
    return True


def api_json(
    clients: ThreadClients,
    *,
    model: str,
    prompt: str,
    retries: int,
    seed: str,
    validator: Any,
) -> tuple[Any, str, int]:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = clients.get().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=6000,
            )
            raw = (response.choices[0].message.content or "").strip()
            value = extract_json_obj(raw)
            if validator(value):
                return value, raw, attempt
            last_error = "response was not valid JSON for the requested schema"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            delay = min(2 ** (attempt - 1) + random.Random(f"{seed}:{attempt}").random(), 20)
            time.sleep(delay)
    raise RuntimeError(last_error)


def mine_one(
    args: argparse.Namespace,
    path: Path,
    skill_bank: str,
    template: str,
    clients: ThreadClients,
) -> str:
    index = path.name.split(".", 1)[0]
    output = args.output / "teacher_analysis" / f"{index}.json"
    raw_output = args.output / "teacher_analysis/raw" / f"{index}.txt"
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids = {str(row["id"]) for row in rows}
    trajectories = "\n\n".join(trajectory_text(row, args.max_reasoning_chars) for row in rows)
    prompt = fill_template(template, skill_bank=skill_bank, trajectories=trajectories)
    if output.is_file():
        return f"{index}: cached"
    value, raw, attempts = api_json(
        clients,
        model=args.model,
        prompt=prompt,
        retries=args.retries,
        seed=index,
        validator=lambda candidate: valid_mining(candidate, ids),
    )
    value["batch_id"] = index
    value["trajectory_ids"] = sorted(ids)
    value["api_attempts"] = attempts
    value["analysis_model"] = args.model
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.write_text(raw + "\n", encoding="utf-8")
    write_json(output, value)
    return f"{index}: complete"


def mine(args: argparse.Namespace) -> None:
    skill_bank = extract_section(args.s0.read_text(encoding="utf-8"), "Planning Skills", "How to Use the Skills")
    template = args.template.read_text(encoding="utf-8")
    inputs = sorted((args.output / "teacher_analysis/batches").glob("batch_*.input.json"))
    if not inputs:
        raise SystemExit("No prepared batches; run prepare first")
    completed_names = {
        path.stem
        for path in (args.output / "teacher_analysis").glob("batch_*.json")
    }
    for path in sorted((args.output / "teacher_analysis").glob("batch_*.json")):
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("analysis_model") != args.model:
            raise SystemExit(f"Cached analysis model differs from this run: {path}")
    pending = [path for path in inputs if path.name.split(".", 1)[0] not in completed_names]
    if args.target_reports:
        needed = max(0, args.target_reports - len(completed_names))
        pending = pending[:needed]
    if not pending:
        print(f"Mining target already satisfied: {len(completed_names)} reports", flush=True)
    else:
        clients = ThreadClients(args.timeout)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    mine_one,
                    args,
                    path,
                    skill_bank,
                    template,
                    clients,
                ): path
                for path in pending
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    print(future.result(), flush=True)
                except Exception as exc:
                    print(f"{path.name}: FAILED {type(exc).__name__}: {exc}", flush=True)
    completed = sorted((args.output / "teacher_analysis").glob("batch_*.json"))
    expected = args.target_reports or len(inputs)
    if len(completed) != expected:
        raise SystemExit(
            f"Mining report count mismatch: found {len(completed)}, expected {expected}; "
            "use a clean output directory for a different sampling configuration"
        )
    manifest = {
        "analysis_model": args.model,
        "analysis_prompt": str(args.template.resolve()),
        "skill_prompt": str(args.s0.resolve()),
        "sample_count": int(json.loads((args.output / "selection.json").read_text())["sample_count"]),
        "batch_size": args.batch_size,
        "report_count": len(completed),
        "seed": args.seed,
    }
    write_json(args.output / "analysis_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    if args.command in ("prepare", "all"):
        prepare(args)
    if args.command in ("mine", "all"):
        mine(args)


if __name__ == "__main__":
    main()
