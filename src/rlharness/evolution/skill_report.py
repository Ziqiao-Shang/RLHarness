#!/usr/bin/env python3
"""Build an accuracy-and-skill-usage report for model-based candidate selection."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SKILL_HEADING_RE = re.compile(r"(?m)^\s*(?:Skill\s+)?(\d+)\s*[.:)]\s+")
SKILL_MARKER_RE = re.compile(r"\[Using Skill\s+S?(\d+)\]", re.IGNORECASE)
SELECTION_SECTION_RE = re.compile(
    r"\[(?:Relevant Strategy Selection|Skill Selection)\](.*?)"
    r"(?=\n\s*\[[^\]\n]+\]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
SKILL_REFERENCE_RE = re.compile(r"\bSkill\s+S?(\d+)\b", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def skill_ids(candidate_dir: Path) -> list[int]:
    path = candidate_dir / "planning_skills.txt"
    if not path.is_file():
        raise SystemExit(f"Missing candidate Skill Bank: {path}")
    return sorted({int(value) for value in SKILL_HEADING_RE.findall(path.read_text(encoding="utf-8"))})


def response_text(row: dict[str, Any]) -> str:
    return str(
        row.get("prediction")
        or row.get("response")
        or row.get("reasoning")
        or row.get("visible_reasoning")
        or ""
    )


def summarize_usage(rows: list[dict[str, Any]], known_ids: list[int]) -> dict[str, Any]:
    selected_counts: Counter[int] = Counter()
    marked_trajectory_counts: Counter[int] = Counter()
    observed_counts: Counter[int] = Counter()
    marker_counts: Counter[int] = Counter()
    correct_counts: Counter[int] = Counter()
    trajectories_with_any_skill = 0
    total_distinct_skills = 0

    for row in rows:
        text = response_text(row)
        selection_match = SELECTION_SECTION_RE.search(text)
        selected = (
            {int(value) for value in SKILL_REFERENCE_RE.findall(selection_match.group(1))}
            if selection_match
            else set()
        )
        markers = [int(value) for value in SKILL_MARKER_RE.findall(text)]
        marked = set(markers)
        used = selected | marked
        if used:
            trajectories_with_any_skill += 1
        total_distinct_skills += len(used)
        marker_counts.update(markers)
        selected_counts.update(selected)
        marked_trajectory_counts.update(marked)
        observed_counts.update(used)
        exact = bool(row.get("format_ok")) and float(row.get("all_acc") or 0.0) >= 1.0
        if exact:
            correct_counts.update(used)

    count = len(rows)
    denominator = max(count, 1)
    observed_ids = set(selected_counts) | set(marked_trajectory_counts) | set(marker_counts)
    per_skill = []
    for skill_id in sorted(set(known_ids) | observed_ids):
        used_count = observed_counts[skill_id]
        per_skill.append(
            {
                "skill_id": skill_id,
                "selected_trajectories": selected_counts[skill_id],
                "marked_trajectories": marked_trajectory_counts[skill_id],
                "observed_trajectories": used_count,
                "usage_rate": used_count / denominator,
                # Compatibility aliases: usage means selection/marker union.
                "trajectory_count": used_count,
                "trajectory_rate": used_count / denominator,
                "marker_count": marker_counts[skill_id],
                "correct_trajectories": correct_counts[skill_id],
                "incorrect_trajectories": used_count - correct_counts[skill_id],
                "exact_rate_when_used": (
                    correct_counts[skill_id] / used_count if used_count else None
                ),
            }
        )

    return {
        "trajectory_count": count,
        "trajectories_with_any_skill": trajectories_with_any_skill,
        "any_skill_usage_rate": trajectories_with_any_skill / denominator,
        "average_distinct_skills_per_trajectory": total_distinct_skills / denominator,
        "unknown_skill_ids": sorted(observed_ids - set(known_ids)),
        "per_skill": per_skill,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    evaluation_keys: set[tuple[str, str, str, str]] = set()
    for evaluation_dir in sorted(args.evaluation_root.glob("candidate_*")):
        summary_path = evaluation_dir / "predictions.summary.json"
        predictions_path = evaluation_dir / "predictions.jsonl"
        evaluation_manifest_path = evaluation_dir / "evaluation_manifest.json"
        candidate_dir = args.candidate_root / evaluation_dir.name
        if not summary_path.is_file() or not predictions_path.is_file():
            continue
        if not evaluation_manifest_path.is_file():
            raise SystemExit(f"Missing evaluation provenance: {evaluation_manifest_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluation_manifest = json.loads(
            evaluation_manifest_path.read_text(encoding="utf-8")
        )
        full_prompt = candidate_dir / "full_prompt.txt"
        evaluated_prompt = Path(str(evaluation_manifest.get("prompt") or "")).resolve()
        if evaluated_prompt != full_prompt.resolve():
            raise SystemExit(f"Evaluated prompt does not match current candidate: {candidate_dir}")
        evaluation_keys.add(
            (
                str(evaluation_manifest.get("domain") or ""),
                str(evaluation_manifest.get("split") or ""),
                str(evaluation_manifest.get("model") or ""),
                str(Path(str(evaluation_manifest.get("sample_ids") or "")).resolve()),
            )
        )
        rows = read_jsonl(predictions_path)
        candidates.append(
            {
                "candidate": evaluation_dir.name,
                "prompt_template": summary.get("prompt_template"),
                "count": int(summary.get("count") or 0),
                "hard_correct": int(summary.get("hard_correct") or 0),
                "exact_accuracy": (
                    int(summary.get("hard_correct") or 0) / max(int(summary.get("count") or 0), 1)
                ),
                "all_acc": float(summary.get("all_acc") or 0.0),
                "part_acc": float(summary.get("part_acc") or 0.0),
                "format_rate": float(summary.get("format_rate") or 0.0),
                "truncated": int(summary.get("truncated") or 0),
                "skill_usage": summarize_usage(rows, skill_ids(candidate_dir)),
                "summary_path": str(summary_path.resolve()),
                "predictions_path": str(predictions_path.resolve()),
                "evaluation_manifest": str(evaluation_manifest_path.resolve()),
            }
        )

    if len(candidates) != 3:
        raise SystemExit(f"Expected exactly 3 completed candidates, found {len(candidates)}")
    counts = {candidate["count"] for candidate in candidates}
    if len(counts) != 1 or next(iter(counts)) == 0:
        raise SystemExit(f"Candidate validation counts do not match: {sorted(counts)}")
    if len(evaluation_keys) != 1:
        raise SystemExit("Candidates were not evaluated with one identical fixed evaluation setup")

    payload = {
        "decision_mode": "gpt-5.6-sol",
        "selected_candidate": None,
        "selection_guidance": [
            "Compare exact accuracy, partial accuracy, format rate, and truncation together.",
            "Inspect whether each candidate actually invokes its skills and whether usage is concentrated or balanced.",
            "Prioritize actual Skill use, while treating correctness and stability metrics as necessary tradeoff evidence.",
            "The configured selector returns one candidate and a structured reason without test-set access.",
        ],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate selection report: {args.output}")


if __name__ == "__main__":
    main()
