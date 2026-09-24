"""Validation and atomic files for initial Prompt construction."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rlharness.common.skill_format import format_skills_block

MUTABLE = ("Planning Skills", "How to Use the Skills", "Few-Shot Demonstrations")
SLOTS = ("{initial_skills}", "{initial_usage}", "{initial_examples}")
FIELDS = ("title", "when_to_apply", "procedure", "check")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_frame(text: str) -> str:
    """Remove only the three mutable bodies, retaining all fixed text verbatim."""
    for heading, slot in zip(MUTABLE, SLOTS):
        pattern = re.compile(r"(^# " + re.escape(heading) + r"\n).*?(?=^# |\Z)", re.M | re.S)
        text, count = pattern.subn(lambda match: match[1] + slot + "\n\n", text)
        if count != 1:
            raise ValueError(f"Expected exactly one section: {heading}")
    return text


def validate_bank(value: Any, *, initial: bool = False) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("skills"), list):
        raise ValueError("Expected skills array")
    skills = value["skills"]
    if not skills:
        raise ValueError("Skill bank must not be empty")
    for number, skill in enumerate(skills, 1):
        if not isinstance(skill, dict) or skill.get("skill_id") != number:
            raise ValueError("Skill IDs must be consecutive from 1")
        for field in FIELDS:
            if not isinstance(skill.get(field), str) or not skill[field].strip():
                raise ValueError(f"Skill {number}: missing {field}")
            if "\n# " in skill[field] or "{" in skill[field] or "}" in skill[field]:
                raise ValueError("Skill text contains reserved prompt syntax")
        if skill.get("category") not in ("image", "table", "fusion"):
            raise ValueError("Skill category must be image, table, or fusion")
    if len({item["title"] for item in skills}) != len(skills):
        raise ValueError("Duplicate skill titles")
    if initial and Counter(item["category"] for item in skills) != {
        "image": 5, "table": 5, "fusion": 5,
    }:
        raise ValueError("Initial map bank requires 5 image + 5 table + 5 fusion skills")
    return value


def assemble(frame: str, bank: dict, usage: str, examples: list[dict]) -> str:
    validate_bank(bank)
    if not isinstance(usage, str) or not usage.strip():
        raise ValueError("Empty how_to_use")
    if "[Relevant Strategy Selection]" not in usage or "[Using Skill N]" not in usage:
        raise ValueError("How-to must require selection and inline Using Skill markers")
    if re.search(r"^# ", usage, re.M) or "{" in usage or "}" in usage:
        raise ValueError("How-to contains reserved prompt syntax")
    bodies = (
        format_skills_block(bank).split("\n", 1)[1].strip(), usage.strip(),
        "\n\n".join(item["text"] for item in examples),
    )
    result = frame
    for slot, body in zip(SLOTS, bodies):
        if frame.count(slot) != 1:
            raise ValueError(f"Missing or duplicate frame slot: {slot}")
        result = result.replace(slot, body)
    if extract_frame(result) != frame:
        raise ValueError("Immutable task frame changed")
    # Check that generated text did not introduce unexpected .format placeholders.
    result.format(question="Question", w1=1, w2=0, w3=0, w4=0)
    return result
