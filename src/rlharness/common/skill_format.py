"""Format a plain numbered skill list for prompt injection (NOT SkillRL schema)."""

from __future__ import annotations

from typing import Any

# Frozen benchmark prompt content. Keep its original language for reproducibility.
SKILL_PROMPT_INTRO = (
    "请你借鉴下面的 skill，更好的完成任务。"
    "请在 <think> 中完成推理，并在 <response> 中输出最终路线。"
    "完整输出（<think> + <response>）合计不超过 2048 token。"
)


def _structured_skill_text(item: dict[str, Any]) -> str:
    when = str(item.get("when_to_apply") or "").strip()
    procedure = str(item.get("procedure") or item.get("principle") or "").strip()
    check = str(item.get("check") or "").strip()
    if not any((when, procedure, check)):
        return ""
    title = str(item.get("title") or "").strip()
    parts = [title] if title else []
    if when:
        parts.append(f"When to apply: {when}")
    if procedure:
        parts.append(f"Procedure: {procedure}")
    if check:
        parts.append(f"Check: {check}")
    return "\n   ".join(parts)


def normalize_skill_list(skill_bank: Any) -> list[str]:
    """Accept {\"skills\": [...]} or a bare JSON list of strings."""
    if skill_bank is None:
        return []
    if isinstance(skill_bank, list):
        items = skill_bank
    elif isinstance(skill_bank, dict):
        items = skill_bank.get("skills")
        if items is None:
            # Legacy SkillRL bank → flatten text for backward compatibility
            items = []
            for s in skill_bank.get("general_skills") or []:
                text = s.get("principle") or s.get("title") or ""
                if text:
                    items.append(str(text).strip())
            for cat, lst in (skill_bank.get("task_specific_skills") or {}).items():
                for s in lst or []:
                    text = s.get("principle") or s.get("title") or ""
                    if text:
                        items.append(str(text).strip())
            for m in skill_bank.get("common_mistakes") or []:
                text = m.get("how_to_avoid") or m.get("description") or ""
                if text:
                    items.append(str(text).strip())
        else:
            items = items
    else:
        return []
    out: list[str] = []
    for x in items:
        if isinstance(x, str):
            t = x.strip()
        elif isinstance(x, dict):
            t = _structured_skill_text(x) or str(
                x.get("text") or x.get("principle") or x.get("title") or ""
            ).strip()
        else:
            t = str(x).strip()
        if t:
            out.append(t)
    return out


def skill_bank_dict(skills: list[str], *, k: int | None = None) -> dict[str, Any]:
    skills = [s.strip() for s in skills if str(s).strip()]
    if k is not None and k > 0:
        skills = skills[:k]
        while len(skills) < k:
            skills.append(f"Re-check weighted trade-offs carefully (pad {len(skills) + 1}).")
    return {"skills": skills, "k": len(skills)}


def format_skills_block(skill_bank: Any) -> str:
    """Numbered list embedded into the official MapTab prompt."""
    skills = normalize_skill_list(skill_bank)
    if not skills:
        return ""
    lines = ["# Planning Skills", ""]
    for i, s in enumerate(skills, 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def count_skills(skill_bank: Any) -> int:
    return len(normalize_skill_list(skill_bank))
