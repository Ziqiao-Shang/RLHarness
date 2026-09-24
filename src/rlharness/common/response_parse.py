"""Parse route-planning model outputs without tool tags."""

from __future__ import annotations

import re
from dataclasses import dataclass


THINK_ANSWER_RE = re.compile(
    r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_ONLY_RE = re.compile(
    r"<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)
# Output format without explicit skill markers.
THINK_RESPONSE_RE = re.compile(
    r"<think>(.*?)</think>\s*<response>(.*?)</response>",
    re.DOTALL | re.IGNORECASE,
)
# Skill-annotated format: think -> skill_retrieve -> response.
THINK_SKILL_RESPONSE_RE = re.compile(
    r"<think>(.*?)</think>\s*<skill_retrieve>(.*?)</skill_retrieve>\s*<response>(.*?)</response>",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
REASONING_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE
)
REASONING_RESPONSE_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>\s*<response>(.*?)</response>",
    re.DOTALL | re.IGNORECASE,
)
SKILL_RETRIEVE_RE = re.compile(
    r"<skill_retrieve>(.*?)</skill_retrieve>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedOutput:
    think: str | None
    final: str | None
    format_ok: bool
    kind: str  # answer | response | skill_response | unknown
    skill_retrieve: str | None = None


def has_think_tags(text: str) -> bool:
    return bool(THINK_RE.search(text or ""))


def extract_think(text: str) -> str | None:
    m = THINK_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_reasoning(text: str) -> str | None:
    m = REASONING_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> str | None:
    text = text or ""

    # Thinking may quote the requested output format or contain provisional
    # answers. Only content after the final closing think tag is a final answer.
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        closing_tags = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))
        if not closing_tags:
            return None
        text = text[closing_tags[-1].end() :]

    matches = list(ANSWER_RE.finditer(text))
    return matches[-1].group(1).strip() if matches else None


def extract_response(text: str) -> str | None:
    m = RESPONSE_RE.search(text or "")
    return m.group(1).strip() if m else None


def extract_skill_retrieve(text: str) -> str | None:
    m = SKILL_RETRIEVE_RE.search(text or "")
    return m.group(1).strip() if m else None


def parse_answer_only(text: str) -> ParsedOutput:
    """Stage1: require <answer>; optional surrounding text / think is ignored for format_ok."""
    text = (text or "").strip()
    ans = extract_answer(text)
    think = extract_think(text)
    return ParsedOutput(
        think=think,
        final=ans,
        format_ok=ans is not None,
        kind="answer",
    )


def parse_think_answer(text: str) -> ParsedOutput:
    """Legacy helper; Stage1 should use parse_answer_only."""
    return parse_answer_only(text)


def parse_think_response(text: str, *, require_skill_retrieve: bool = False) -> ParsedOutput:
    """
    Parse a route-planning response.
    - require_skill_retrieve=False (format warmup): need <think> + <response>
    - require_skill_retrieve=True (skill phase): need <think> + <skill_retrieve> + <response>
    """
    text = (text or "").strip()
    m3 = THINK_SKILL_RESPONSE_RE.search(text)
    if m3:
        return ParsedOutput(
            think=m3.group(1).strip(),
            skill_retrieve=m3.group(2).strip(),
            final=m3.group(3).strip(),
            format_ok=True,
            kind="skill_response",
        )
    think = extract_think(text)
    skill = extract_skill_retrieve(text)
    resp = extract_response(text)
    if require_skill_retrieve:
        ok = think is not None and skill is not None and resp is not None
        return ParsedOutput(
            think=think,
            skill_retrieve=skill,
            final=resp,
            format_ok=ok,
            kind="skill_response",
        )
    # Warmup: skill_retrieve optional; format ok if think+response
    m2 = THINK_RESPONSE_RE.fullmatch(text)
    if m2:
        return ParsedOutput(
            think=m2.group(1).strip(),
            final=m2.group(2).strip(),
            format_ok=True,
            kind="response",
            skill_retrieve=skill,
        )
    ok = think is not None and resp is not None
    return ParsedOutput(
        think=think,
        final=resp,
        format_ok=ok,
        kind="response",
        skill_retrieve=skill,
    )


def parse_reasoning_response(text: str) -> ParsedOutput:
    """No-thinking contract: one nonempty reasoning block, then one route."""
    text = (text or "").strip()
    match = REASONING_RESPONSE_RE.fullmatch(text)
    reasoning = extract_reasoning(text)
    response = extract_response(text)
    format_ok = bool(
        match
        and match.group(1).strip()
        and match.group(2).strip()
        and not has_think_tags(text)
    )
    return ParsedOutput(
        think=None,
        final=response,
        format_ok=format_ok,
        kind="reasoning_response",
    )


def wrap_answer(answer: str) -> str:
    """Stage1 SFT/RL target: answer tags only."""
    return f"<answer>{str(answer).strip()}</answer>"


def wrap_think_answer(think: str, answer: str) -> str:
    """Deprecated for Stage1; kept for compatibility."""
    _ = think
    return wrap_answer(answer)


def wrap_think_response(think: str, response: str) -> str:
    return f"<think>{think.strip()}</think>\n<response>{str(response).strip()}</response>"


def wrap_think_skill_response(think: str, skill_retrieve: str, response: str) -> str:
    return (
        f"<think>{think.strip()}</think>\n"
        f"<skill_retrieve>{skill_retrieve.strip()}</skill_retrieve>\n"
        f"<response>{str(response).strip()}</response>"
    )
