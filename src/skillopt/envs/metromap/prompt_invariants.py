from __future__ import annotations

import re


REQUIRED_PLACEHOLDERS = ("{question}", "{w1}", "{w2}", "{w3}", "{w4}")


def _section(document: str, heading: str, next_heading: str | None) -> str:
    start = document.find(heading)
    if start < 0:
        return ""
    if next_heading is None:
        return document[start:].strip()
    end = document.find(next_heading, start + len(heading))
    if end < 0:
        return ""
    return document[start:end].strip()


def _skill_ids(document: str) -> list[int]:
    planning = _section(document, "# Planning Skills", "# How to Use the Skills")
    return [
        int(match.group(1))
        for match in re.finditer(r"(?m)^(\d+)\.\s+\S", planning)
    ]


def validate_prompt_candidate(candidate: str, baseline: str) -> tuple[bool, list[str]]:
    """Enforce the experiment's immutable prompt contract.

    Skill descriptions, reasoning guidance, examples, and ordinary task wording
    remain editable. The scoring semantics and interface contract do not.
    """
    reasons: list[str] = []

    for placeholder in REQUIRED_PLACEHOLDERS:
        expected = baseline.count(placeholder)
        actual = candidate.count(placeholder)
        if actual != expected:
            reasons.append(
                f"placeholder {placeholder} count changed: {expected} -> {actual}"
            )

    baseline_ids = _skill_ids(baseline)
    candidate_ids = _skill_ids(candidate)
    if candidate_ids != baseline_ids:
        reasons.append(
            "skill numbering/order changed: "
            f"expected {baseline_ids}, got {candidate_ids}"
        )

    protected_sections = (
        (
            "authoritative scoring rules",
            "# Authoritative Scoring Rules",
            "# Planning Skills",
        ),
        (
            "skill usage contract",
            "# How to Use the Skills",
            "# Required Procedure",
        ),
        (
            "output and route format",
            "# Output Format Constraints",
            None,
        ),
    )
    for label, heading, next_heading in protected_sections:
        expected = _section(baseline, heading, next_heading)
        actual = _section(candidate, heading, next_heading)
        if not expected:
            reasons.append(f"baseline is missing protected section: {label}")
        elif actual != expected:
            reasons.append(f"protected section changed: {label}")

    return not reasons, reasons
