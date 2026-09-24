#!/usr/bin/env python3
"""Prune zero or extremely low-use numbered skills from a rollout prompt."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SKILL_BLOCK = re.compile(r"(?ms)^(\d+)\.\s+.*?(?=^\d+\.\s+|\Z)")
MARKER = re.compile(r"\[Using\s+Skill\s+(?:S)?(\d+)\]", re.IGNORECASE)
SELECTION_SECTION = re.compile(
    r"\[(?:Relevant Strategy Selection|Skill Selection)\](.*?)"
    r"(?=\n\s*\[[^\]\n]+\]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
SKILL_REFERENCE = re.compile(r"\bSkill\s+(?:S)?(\d+)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-prompt", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--min-usage-rate",
        type=float,
        default=0.005,
        help="Keep skills used in at least this fraction of trajectories (default: 0.005).",
    )
    parser.add_argument("--keep", default="", help="Comma-separated skill IDs that must remain.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def response_text(row: dict[str, Any]) -> str:
    return str(
        row.get("prediction")
        or row.get("response")
        or row.get("reasoning")
        or row.get("visible_reasoning")
        or ""
    )


def main() -> None:
    args = parse_args()
    prompt = args.prompt.read_text(encoding="utf-8")
    prefix, remainder = prompt.split("# Planning Skills\n", 1)
    skill_section, suffix = remainder.split("\n# How to Use the Skills\n", 1)
    blocks = [(match.group(1), match.group(0).strip()) for match in SKILL_BLOCK.finditer(skill_section.strip())]
    if not blocks:
        raise SystemExit("No numbered skills found")

    rows = load_jsonl(args.rollouts)
    selected_counts = {skill_id: 0 for skill_id, _ in blocks}
    marked_counts = {skill_id: 0 for skill_id, _ in blocks}
    observed_counts = {skill_id: 0 for skill_id, _ in blocks}
    for row in rows:
        text = response_text(row)
        match = SELECTION_SECTION.search(text)
        selected = set(SKILL_REFERENCE.findall(match.group(1))) if match else set()
        marked = set(MARKER.findall(text))
        observed = selected | marked
        for skill_id in observed_counts:
            selected_counts[skill_id] += int(skill_id in selected)
            marked_counts[skill_id] += int(skill_id in marked)
            observed_counts[skill_id] += int(skill_id in observed)
    if not any(observed_counts.values()):
        raise SystemExit(
            "No selected or explicitly marked skills found; refusing to prune an unobservable bank"
        )

    forced = {item.strip() for item in args.keep.split(",") if item.strip()}
    total = len(rows)
    removed: list[str] = []
    kept: list[str] = []
    statistics = []
    for skill_id, _ in blocks:
        count = observed_counts[skill_id]
        rate = count / total if total else 0.0
        low = rate < args.min_usage_rate
        decision = "keep" if skill_id in forced or not low else "remove"
        (kept if decision == "keep" else removed).append(skill_id)
        statistics.append(
            {
                "skill_id": skill_id,
                "selected_trajectories": selected_counts[skill_id],
                "marked_trajectories": marked_counts[skill_id],
                "observed_trajectories": count,
                "used_trajectories": count,
                "total_trajectories": total,
                "usage_rate": rate,
                "decision": decision,
                "forced_keep": skill_id in forced,
            }
        )
    if len(kept) < 3:
        raise SystemExit(f"Pruning would leave only {len(kept)} skills; adjust thresholds or --keep")

    retained = "\n".join(block for skill_id, block in blocks if skill_id in kept)
    # Keep old markers as semantic evidence for the later API rewrite stage.
    result = prefix + "# Planning Skills\n\n" + retained + "\n\n# How to Use the Skills\n" + suffix
    args.output_prompt.parent.mkdir(parents=True, exist_ok=True)
    args.output_prompt.write_text(result, encoding="utf-8")
    report = {
        "source_prompt": str(args.prompt.resolve()),
        "source_rollouts": str(args.rollouts.resolve()),
        "total_trajectories": total,
        "thresholds": {"min_usage_rate": args.min_usage_rate},
        "kept_skill_ids": kept,
        "removed_skill_ids": removed,
        "statistics": statistics,
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"kept={kept} removed={removed} output={args.output_prompt}")


if __name__ == "__main__":
    main()
