"""Build route-planning prompts from MapTab shortest_path_csv_vertex2 / only_vertex2.

Uses the official MapTab planning prompt filled like `csv_vertex2`.

When skills are enabled for MetroMap, use the dedicated skill-augmented prompt
and insert the numbered skill list at its explicit placeholder. Other domains
retain the legacy skill-injection behavior.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from rlharness.common.config import ROOT, maptab_root
from rlharness.common.skill_format import (
    SKILL_PROMPT_INTRO,
    format_skills_block,
    normalize_skill_list,
)

TASK_TEMPLATE_ROOT = ROOT / "prompts"

OFFICIAL_PROMPT_NAME = {
    "metromap": "common/task_base.txt",
    "travelmap": "travelmap_shortest_path_with_constraint_1_2_3_4_only_vertex2.txt",
}

SKILL_PROMPT_NAME = {
    "metromap": "common/skill_injection.txt",
    "travelmap": "travelmap_shortest_path_with_constraint_1_2_3_4_only_vertex2_skill_augmented.txt",
}

TABLE_INTRO = {
    "metromap": "This is a vertex table of a subway map.",
    "travelmap": "This is a vertex table of a scenic area planning map.",
}

IMAGE_INTRO = {
    "metromap": "This is the subway map image.",
    "travelmap": "This is the scenic area planning map image.",
}

_RULES_HEADING = re.compile(r"(?m)^# Attributes Scaling Rules\s*$")
_EDGE_TABLE_IN_QUESTION = re.compile(
    r",\s*Edge Table\s+and\s+Vertex Table",
    flags=re.IGNORECASE,
)
_THINKING_OUTPUT_RULE = re.compile(
    r"(?m)^- Reason step-by-step inside `<think>\.\.\.</think>`.*\n?"
)
_SKILLS_PLACEHOLDER = "__PLANNING_SKILLS__"


@lru_cache(maxsize=16)
def _load_official_prompt(domain: str, skill_augmented: bool = False) -> str:
    name = SKILL_PROMPT_NAME.get(domain) if skill_augmented else None
    name = name or OFFICIAL_PROMPT_NAME[domain]
    local = TASK_TEMPLATE_ROOT / name
    if local.exists():
        return local.read_text(encoding="utf-8").strip()
    remote = maptab_root() / domain / "prompts" / name
    if remote.exists():
        return remote.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(
        f"Official planning prompt missing for {domain}: tried {local} and {remote}"
    )


def _weights_tuple(rec: dict[str, Any]) -> tuple[float, float, float, float]:
    w = rec.get("weights") or [0.25, 0.25, 0.25, 0.25]
    if len(w) != 4:
        raise ValueError(f"Expected 4 weights, got {w!r} for {rec.get('sample_id')}")
    return float(w[0]), float(w[1]), float(w[2]), float(w[3])


def resolve_image(rel: str | None) -> str | None:
    if not rel:
        return None
    p = maptab_root() / rel
    return str(p) if p.exists() else None


def resolve_vertex_table_path(rec: dict[str, Any]) -> Path | None:
    rel = rec.get("vertex_tab") or rec.get("vertex2_tab")
    if not rel:
        return None
    root = maptab_root()
    json_path = root / rel
    csv_path = root / str(rel).replace(".json", ".csv")
    if csv_path.exists():
        return csv_path
    if json_path.exists():
        return json_path
    return None


def load_table_text(
    rel: str | None = None,
    *,
    max_chars: int = 12000,
    rec: dict[str, Any] | None = None,
) -> str:
    path: Path | None = None
    if rec is not None:
        path = resolve_vertex_table_path(rec)
    elif rel:
        path = resolve_vertex_table_path({"vertex_tab": rel})
    if path is None or not path.exists():
        return ""
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        text = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _inject_skills(prompt_body: str, skill_block: str) -> str:
    block = skill_block.strip()
    if not block:
        return prompt_body
    m = _RULES_HEADING.search(prompt_body)
    if not m:
        return prompt_body.rstrip() + "\n\n" + block + "\n"
    return prompt_body[: m.start()] + "\n" + block + "\n\n" + prompt_body[m.start() :]


def normalize_question(question: str) -> str:
    """Keep source questions aligned with the map + Vertex Table inputs."""
    return _EDGE_TABLE_IN_QUESTION.sub(" and Vertex Table", question.strip())


def _set_output_mode(prompt_body: str, *, thinking: bool) -> str:
    if thinking:
        return prompt_body
    body = _THINKING_OUTPUT_RULE.sub("", prompt_body)
    body = body.replace(
        "Inside `<think>`, follow this procedure concisely:",
        "Before producing the response, internally follow this procedure concisely:",
    )
    return body.replace(
        "Put the final route inside `<response>...</response>`",
        "Do not output `<think>` tags. Put the final route directly inside "
        "`<response>...</response>`",
    )


def fill_official_prompt(
    rec: dict[str, Any],
    *,
    thinking: bool = True,
    skill_augmented: bool = False,
    prompt_template: str | None = None,
) -> str:
    domain = rec.get("domain") or "metromap"
    if domain not in OFFICIAL_PROMPT_NAME:
        raise ValueError(f"Unsupported planning domain: {domain}")
    tmpl = prompt_template or _load_official_prompt(domain, skill_augmented)
    w1, w2, w3, w4 = _weights_tuple(rec)
    body = tmpl.format(
        question=normalize_question(str(rec.get("question", ""))),
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
    )
    return _set_output_mode(body, thinking=thinking)


def build_instruction(
    rec: dict[str, Any],
    *,
    skill_bank: Any = None,
    include_skills: bool = False,
    thinking: bool = True,
    table_max_chars: int = 12000,
    prompt_template: str | None = None,
) -> str:
    domain = rec.get("domain") or "metromap"
    skills = normalize_skill_list(skill_bank) if include_skills else []
    use_dedicated_skill_prompt = bool(skills) and domain in SKILL_PROMPT_NAME
    body = fill_official_prompt(
        rec,
        thinking=thinking,
        skill_augmented=use_dedicated_skill_prompt,
        prompt_template=prompt_template,
    )

    if include_skills and skills:
        skill_block = format_skills_block(skill_bank)
        if use_dedicated_skill_prompt:
            if _SKILLS_PLACEHOLDER not in body:
                raise ValueError("Skill-augmented prompt is missing its skills placeholder.")
            body = body.replace(_SKILLS_PLACEHOLDER, skill_block)
        else:
            body = _inject_skills(body, skill_block)
            body = SKILL_PROMPT_INTRO + "\n\n" + body

    table = load_table_text(rec=rec, max_chars=table_max_chars)
    img = resolve_image(rec.get("figure"))

    parts = [body.strip(), ""]
    parts.append(TABLE_INTRO.get(domain, "This is a vertex table."))
    parts.append(table if table else "(Vertex table missing.)")
    parts.append("")
    parts.append(IMAGE_INTRO.get(domain, "This is the map image."))
    parts.append("(See attached image.)" if img else "(No image.)")
    return "\n".join(parts)


def build_prompt_bundle(
    rec: dict[str, Any],
    *,
    skill_bank: Any = None,
    include_skills: bool = False,
    thinking: bool = True,
    table_max_chars: int = 12000,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    text = build_instruction(
        rec,
        skill_bank=skill_bank,
        include_skills=include_skills,
        thinking=thinking,
        table_max_chars=table_max_chars,
        prompt_template=prompt_template,
    )
    img = resolve_image(rec.get("figure"))
    return {
        "text": text,
        "image_path": img,
        "table_text": load_table_text(rec=rec, max_chars=table_max_chars),
        "table_path": str(p) if (p := resolve_vertex_table_path(rec)) else None,
    }
