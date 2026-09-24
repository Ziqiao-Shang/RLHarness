"""Deterministically audit numeric claims in generated Stage 2 SFT labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from rlharness.common.data_load import load_planning
from rlharness.common.prompt_build import resolve_vertex_table_path


NUMBER_RE = re.compile(
    r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
)
TRANSFER_SUFFIX = "(transfer)"


@dataclass(frozen=True)
class RouteMetrics:
    n: int
    time_sum: float
    transfer_time_sum: float
    price_sum: float
    comfort_cost_sum: float
    reliability_cost_sum: float
    t: float
    p: float
    c: float
    r: float
    score: float
    comfort_count: int | None = None
    reliability_count: int | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_vertex_rows(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {str(row["Vertex"]): row for row in csv.DictReader(handle)}

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("rows") or list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"Unsupported vertex-table JSON structure: {path}")
    return {str(row["Vertex"]): row for row in data if isinstance(row, dict)}


def parse_route(route: str, station_names: set[str]) -> list[tuple[str, bool]]:
    """Parse hyphen-separated routes even when a station name contains a hyphen."""
    candidates: list[tuple[str, str, bool]] = []
    for name in station_names:
        candidates.append((name + TRANSFER_SUFFIX, name, True))
        candidates.append((name, name, False))
    candidates.sort(key=lambda item: len(item[0]), reverse=True)

    @lru_cache(maxsize=None)
    def solve(position: int) -> tuple[tuple[str, bool], ...] | None:
        for token, name, is_transfer in candidates:
            if not route.startswith(token, position):
                continue
            end = position + len(token)
            if end == len(route):
                return ((name, is_transfer),)
            if route[end : end + 1] != "-":
                continue
            suffix = solve(end + 1)
            if suffix is not None:
                return ((name, is_transfer),) + suffix
        return None

    parsed = solve(0)
    if parsed is None:
        raise ValueError(f"Could not parse route using the vertex table: {route}")
    return list(parsed)


def compute_route_metrics(
    route: str,
    table: dict[str, dict[str, Any]],
    weights: list[float],
    *,
    domain: str = "metromap",
) -> RouteMetrics:
    parsed = parse_route(route, set(table))
    if len(parsed) < 2:
        raise ValueError("Route must contain at least a start and destination")

    if domain == "travelmap":
        scored = parsed
        n = len(scored)
        time_sum = sum(float(table[name]["Time"]) for name, _ in scored)
        price_sum = sum(
            0.0
            if index < n - 1 and float(table[name]["Time"]) == 0.0
            else float(table[name]["Price"])
            for index, (name, _) in enumerate(scored)
        )
        comfort_values = [
            float(table[name]["Comfort Level"])
            for name, _ in scored
            if float(table[name]["Comfort Level"]) != 0.0
        ]
        reliability_values = [
            float(table[name]["Reliability"])
            for name, _ in scored
            if float(table[name]["Reliability"]) != 0.0
        ]
        if not comfort_values or not reliability_values:
            raise ValueError("TravelMap quality averages require nonzero values")
        comfort_cost_sum = sum(1.0 - value / 5.0 for value in comfort_values)
        reliability_cost_sum = sum(1.0 - value for value in reliability_values)
        transfer_time_sum = 0.0
        t = time_sum / 180.0
        p = price_sum / 200.0
        c = comfort_cost_sum / len(comfort_values)
        r = reliability_cost_sum / len(reliability_values)
        score = sum(
            float(weight) * value
            for weight, value in zip(weights, (t, p, c, r))
        )
        return RouteMetrics(
            n=n,
            time_sum=time_sum,
            transfer_time_sum=transfer_time_sum,
            price_sum=price_sum,
            comfort_cost_sum=comfort_cost_sum,
            reliability_cost_sum=reliability_cost_sum,
            t=t,
            p=p,
            c=c,
            r=r,
            score=score,
            comfort_count=len(comfort_values),
            reliability_count=len(reliability_values),
        )
    if domain != "metromap":
        raise ValueError(f"Unsupported domain: {domain}")

    scored = parsed[:-1]
    n = len(scored)

    def total(column: str) -> float:
        return sum(float(table[name][column]) for name, _ in scored)

    time_sum = total("Time")
    price_sum = total("Price")
    comfort_cost_sum = sum(
        1.0 - float(table[name]["Comfort Level"]) for name, _ in scored
    )
    reliability_cost_sum = sum(
        1.0 - float(table[name]["Reliability"]) for name, _ in scored
    )
    transfer_time_sum = sum(
        float(table[name]["Transfer Time"])
        for name, is_transfer in scored
        if is_transfer
    )
    t = (time_sum + transfer_time_sum) / 3.0
    p = price_sum / 1.5
    c = comfort_cost_sum / n
    r = reliability_cost_sum / n
    score = sum(float(weight) * value for weight, value in zip(weights, (t, p, c, r)))
    return RouteMetrics(
        n=n,
        time_sum=time_sum,
        transfer_time_sum=transfer_time_sum,
        price_sum=price_sum,
        comfort_cost_sum=comfort_cost_sum,
        reliability_cost_sum=reliability_cost_sum,
        t=t,
        p=p,
        c=c,
        r=r,
        score=score,
        comfort_count=n,
        reliability_count=n,
    )


def nearest_numeric_delta(text: str, expected: float) -> tuple[float, float | None]:
    values = [float(match.group(0)) for match in NUMBER_RE.finditer(text)]
    if not values:
        return math.inf, None
    nearest = min(values, key=lambda value: abs(value - expected))
    return abs(nearest - expected), nearest


def audit_claimed_numbers(
    reasoning: str,
    metrics: RouteMetrics,
    *,
    score_tolerance: float,
    component_tolerance: float,
) -> dict[str, Any]:
    expected = {
        "T": metrics.t,
        "P": metrics.p,
        "C": metrics.c,
        "R": metrics.r,
        "Score": metrics.score,
    }
    checks: dict[str, dict[str, Any]] = {}
    for name, value in expected.items():
        delta, nearest = nearest_numeric_delta(reasoning, value)
        tolerance = score_tolerance if name == "Score" else component_tolerance
        checks[name] = {
            "expected": value,
            "nearest_claim": nearest,
            "absolute_error": delta,
            "tolerance": tolerance,
            "passed": delta <= tolerance,
        }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--domain", default="metromap", choices=("metromap", "travelmap"))
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--clean_output", type=Path)
    parser.add_argument("--rejected_output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--score_tolerance", type=float, default=5e-4)
    parser.add_argument("--component_tolerance", type=float, default=5e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = {
        row["sample_id"]: row
        for row in load_planning(args.domain, args.split, only_vertex2=True)
    }
    labels = _read_jsonl(args.input_file)
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    table_cache: dict[Path, dict[str, dict[str, Any]]] = {}

    for label in labels:
        sample_id = str(label.get("sample_id") or "")
        detail: dict[str, Any] = {"sample_id": sample_id, "passed": False}
        try:
            source = source_rows[sample_id]
            table_path = resolve_vertex_table_path(source)
            if table_path is None:
                raise FileNotFoundError(f"Vertex table not found for {sample_id}")
            table = table_cache.setdefault(table_path, _load_vertex_rows(table_path))
            route = str(label.get("gt_route") or source.get("gt_route") or "")
            metrics = compute_route_metrics(
                route,
                table,
                list(source["weights"]),
                domain=args.domain,
            )
            audit = audit_claimed_numbers(
                str(label.get("reasoning") or label.get("think") or ""),
                metrics,
                score_tolerance=args.score_tolerance,
                component_tolerance=args.component_tolerance,
            )
            detail.update(
                {
                    "passed": audit["passed"],
                    "table_path": str(table_path),
                    "metrics": asdict(metrics),
                    **audit,
                }
            )
        except Exception as exc:
            detail["error"] = f"{type(exc).__name__}: {exc}"

        details.append(detail)
        if detail["passed"]:
            clean.append(label)
        else:
            rejected.append({**label, "numeric_audit": detail})

    clean_output = args.clean_output or args.input_file.with_name("labels.numeric_clean.jsonl")
    rejected_output = args.rejected_output or args.input_file.with_name(
        "labels.numeric_rejected.jsonl"
    )
    report_path = args.report or args.input_file.with_name("numeric_audit_report.json")
    _write_jsonl(clean_output, clean)
    _write_jsonl(rejected_output, rejected)
    report_path.write_text(
        json.dumps(
            {
                "input_file": str(args.input_file),
                "total": len(labels),
                "passed": len(clean),
                "rejected": len(rejected),
                "score_tolerance": args.score_tolerance,
                "component_tolerance": args.component_tolerance,
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"audited={len(labels)} passed={len(clean)} rejected={len(rejected)} "
        f"pass_rate={(len(clean) / len(labels) if labels else 0):.2%}"
    )
    print(f"clean: {clean_output}")
    print(f"rejected: {rejected_output}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
