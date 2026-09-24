"""Generate audited Stage 2 skill-SFT labels with concurrent teacher calls."""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rlharness.common.api_client import (
    DEFAULT_MODEL,
    chat_text,
    multimodal_messages,
    openlux_client,
)
from rlharness.common.config import ROOT
from rlharness.common.data_load import load_planning
from rlharness.common.prompt_build import build_prompt_bundle
from rlharness.common.skill_format import format_skills_block, normalize_skill_list


REQUIRED_REASONING_SECTIONS = (
    "[Task and Map Analysis]",
    "[Relevant Strategy Selection]",
    "[Candidate Routes]",
    "[Score Calculation]",
    "[Decision and Verification]",
)

INLINE_MARKER_REASONING_SECTIONS = (
    "[Task and Map Analysis]",
    "[Candidate Routes]",
    "[Score Calculation]",
    "[Decision and Verification]",
)

VALIDATION_PROFILES = ("legacy", "inline_markers")

EXACT_OUTPUT_RE = re.compile(
    r"\s*<reasoning>(.*?)</reasoning>\s*<response>(.*?)</response>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


class ThreadClients:
    """Create one OpenLux client per worker thread."""

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout
        self.local = threading.local()

    def get(self) -> Any:
        client = getattr(self.local, "client", None)
        if client is None:
            client = openlux_client(timeout=self.timeout)
            self.local.client = client
        return client


def _teacher_prompt_path(domain: str) -> Path:
    name = f"data_process/teacher_{domain}.txt"
    path = ROOT / "prompts" / name
    if not path.is_file():
        raise FileNotFoundError(f"Teacher prompt is missing: {path}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lock_generation_manifest(
    path: Path,
    payload: dict[str, Any],
    *,
    artifacts: tuple[Path, ...],
    resume: bool,
) -> None:
    """Prevent resumed teacher generation from mixing different inputs."""
    existing_artifacts = [artifact for artifact in artifacts if artifact.exists()]
    if resume and existing_artifacts and not path.is_file():
        raise SystemExit(
            f"Existing teacher outputs have no generation manifest: {existing_artifacts[0]}. "
            "Use a new SFT_TEACHER_DIR."
        )
    if resume and path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SystemExit(
                f"Teacher generation inputs differ from the existing run: {path}. "
                "Use a new SFT_TEACHER_DIR."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def load_token_counter(tokenizer_path: str | Path) -> Any:
    """Load the exact tokenizer without importing the full Transformers stack."""
    path = Path(tokenizer_path)
    tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json: {tokenizer_json}")

    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_json))


def load_excluded_sample_ids(path: Path | None) -> set[str]:
    """Load sample IDs from a selection JSON, JSON list, or JSONL file."""
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"Sample-ID exclusion file is missing: {path}")

    if path.suffix == ".jsonl":
        return {
            str(row["sample_id"])
            for row in _read_jsonl(path)
            if row.get("sample_id")
        }

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("sample_ids", [])
    if not isinstance(value, list):
        raise ValueError(
            f"Expected a JSON list or an object with sample_ids in {path}"
        )
    return {str(sample_id) for sample_id in value if str(sample_id).strip()}


def select_map_stratified(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Round-robin across shuffled maps so a small pilot is geographically diverse."""
    if limit <= 0 or limit >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("figure") or "(no-image)")
        groups.setdefault(key, []).append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        added = False
        for key in keys:
            group = groups[key]
            if group:
                selected.append(group.pop())
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return selected


def _difficulty_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("Map_Difficulty") or "Medium"),
        str(row.get("Query_Difficulty") or "Medium"),
    )


def select_difficulty_stratified(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Preserve the source difficulty mix while retaining map diversity."""
    if limit <= 0 or limit >= len(rows):
        return list(rows)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_difficulty_key(row), []).append(row)

    # Largest-remainder allocation makes the integer quotas sum exactly to limit.
    exact = {
        key: limit * len(group) / len(rows)
        for key, group in groups.items()
    }
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    remainder_order = sorted(
        groups,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in remainder_order[:remaining]:
        quotas[key] += 1

    selected: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(groups)):
        quota = quotas[key]
        if quota:
            selected.extend(
                select_map_stratified(
                    groups[key],
                    limit=quota,
                    seed=seed + (index + 1) * 1009,
                )
            )

    random.Random(seed).shuffle(selected)
    return selected


def parse_difficulty_quotas(value: str | None) -> dict[tuple[str, str], int]:
    """Parse exact difficulty quotas from JSON text or a JSON file."""
    if not value:
        return {}
    path = Path(value)
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("Difficulty quotas must be a JSON object")

    quotas: dict[tuple[str, str], int] = {}
    for name, count in raw.items():
        parts = re.split(r"[xX*×]", str(name))
        if len(parts) != 2 or any(part not in {"Easy", "Medium", "Hard"} for part in parts):
            raise ValueError(
                f"Invalid difficulty bucket {name!r}; expected MapDifficultyxQueryDifficulty"
            )
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Difficulty quota for {name!r} must be a non-negative integer")
        if count:
            quotas[(parts[0], parts[1])] = count
    if not quotas:
        raise ValueError("At least one difficulty quota must be positive")
    return quotas


def select_difficulty_quotas(
    rows: list[dict[str, Any]],
    *,
    quotas: dict[tuple[str, str], int],
    seed: int,
) -> list[dict[str, Any]]:
    """Select exact difficulty buckets while keeping each bucket map-diverse."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_difficulty_key(row), []).append(row)

    selected: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(quotas)):
        available = groups.get(key, [])
        quota = quotas[key]
        if quota > len(available):
            raise ValueError(
                f"Difficulty bucket {key[0]}x{key[1]} requests {quota} samples "
                f"but only {len(available)} are eligible"
            )
        selected.extend(
            select_map_stratified(
                available,
                limit=quota,
                seed=seed + (index + 1) * 1009,
            )
        )

    random.Random(seed).shuffle(selected)
    return selected


def gt_starts_with_transfer(row: dict[str, Any]) -> bool:
    """Return whether the GT marks the requested start station as a transfer."""
    start = str(row.get("station_1") or "").strip()
    route = str(row.get("gt_route") or "").strip()
    return bool(start) and route.startswith(f"{start}(transfer)")


def validate_teacher_output(
    raw: str,
    *,
    gt_route: str,
    tokenizer: Any,
    min_reasoning_tokens: int,
    max_output_tokens: int,
    validation_profile: str = "legacy",
) -> dict[str, Any]:
    match = EXACT_OUTPUT_RE.fullmatch(raw or "")
    if not match:
        if re.match(r"\s*<reasoning>", raw or "", flags=re.IGNORECASE) and not re.search(
            r"</reasoning>\s*<response>.*?</response>\s*$",
            raw or "",
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raise ValueError(
                "output was truncated before the closing tags and final route; "
                "regenerate more concisely and reserve enough space for the complete "
                f"<response> within the {max_output_tokens}-token output limit"
            )
        raise ValueError(
            "output must contain exactly one <reasoning> block and one <response> block"
        )

    reasoning = match.group(1).strip()
    route = match.group(2).strip()
    if not reasoning:
        raise ValueError("the <reasoning> block is empty")
    if route != gt_route:
        raise ValueError("the route inside <response> does not exactly match the ground truth")
    if re.search(r"<skill_retrieve>|tool[_ -]?call", raw, flags=re.IGNORECASE):
        raise ValueError("tool or skill-retrieval syntax is forbidden")
    if re.search(
        r"ground[- ]truth|according to (?:the )?gt\b|because (?:the )?gt\b",
        reasoning,
        flags=re.IGNORECASE,
    ):
        raise ValueError("the reasoning explicitly leaks the ground truth")

    if validation_profile not in VALIDATION_PROFILES:
        raise ValueError(f"unknown validation profile: {validation_profile}")

    required_sections = (
        INLINE_MARKER_REASONING_SECTIONS
        if validation_profile == "inline_markers"
        else REQUIRED_REASONING_SECTIONS
    )
    missing_sections = [section for section in required_sections if section not in reasoning]
    if missing_sections:
        raise ValueError(f"missing reasoning sections: {', '.join(missing_sections)}")

    section_positions = [reasoning.index(section) for section in required_sections]
    if section_positions != sorted(section_positions):
        raise ValueError("reasoning sections are not in the required order")

    if validation_profile == "inline_markers":
        if "[Relevant Strategy Selection]" in reasoning:
            raise ValueError(
                "inline-marker reasoning must not contain a skill-selection stage"
            )
        raw_markers = re.findall(
            r"\[Using Skill\s+([^\]]+)\]",
            reasoning,
            flags=re.IGNORECASE,
        )
        invalid_markers = [
            marker for marker in raw_markers if not re.fullmatch(r"S[1-6]", marker)
        ]
        if invalid_markers:
            raise ValueError(
                "invalid inline skill markers: " + ", ".join(invalid_markers)
            )
        skill_markers = [marker.upper() for marker in raw_markers]
        if len(set(skill_markers)) != len(skill_markers):
            raise ValueError("each inline skill marker may appear at most once")
    else:
        selection_start = reasoning.index("[Relevant Strategy Selection]") + len(
            "[Relevant Strategy Selection]"
        )
        selection_end = reasoning.index("[Candidate Routes]")
        selection = reasoning[selection_start:selection_end]
        selected_skill_ids = re.findall(
            r"(?mi)^\s*-\s*Skill\s+(\d+)\b",
            selection,
        )
        if not 3 <= len(selected_skill_ids) <= 6:
            raise ValueError(
                "the [Relevant Strategy Selection] section must list exactly 3 to 6 skills"
            )
        if len(set(selected_skill_ids)) != len(selected_skill_ids):
            raise ValueError(
                "the [Relevant Strategy Selection] section contains duplicate skills"
            )

        application_text = reasoning[selection_end:]
        missing_applications = [
            skill_id
            for skill_id in selected_skill_ids
            if not re.search(
                rf"\[Using Skill\s+{re.escape(skill_id)}\]",
                application_text,
                flags=re.IGNORECASE,
            )
        ]
        if missing_applications:
            raise ValueError(
                "selected skills missing later application markers: "
                + ", ".join(missing_applications)
            )
        skill_markers = selected_skill_ids

    output = f"<reasoning>{reasoning}</reasoning>\n<response>{route}</response>"
    reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False))
    output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
    if reasoning_tokens < min_reasoning_tokens:
        raise ValueError(
            "reasoning is too short: "
            f"{reasoning_tokens} tokens, minimum is {min_reasoning_tokens}"
        )
    if output_tokens > max_output_tokens:
        raise ValueError(
            f"complete output is too long: {output_tokens} tokens, maximum is {max_output_tokens}"
        )
    return {
        "reasoning": reasoning,
        "route": route,
        "output": output,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "skill_markers": skill_markers,
    }


def generate_one(
    rec: dict[str, Any],
    *,
    skill_bank: Any,
    student_prompt_template: str | None,
    include_skills: bool,
    teacher_template: str,
    teacher_prompt_path: Path,
    clients: ThreadClients,
    tokenizer: Any,
    model: str,
    temperature: float,
    teacher_max_tokens: int,
    min_reasoning_tokens: int,
    max_output_tokens: int,
    retries: int,
    validation_profile: str,
) -> dict[str, Any]:
    started = time.monotonic()
    gt_route = str(rec["gt_route"]).strip()
    student = build_prompt_bundle(
        rec,
        skill_bank=skill_bank,
        include_skills=include_skills,
        thinking=False,
        prompt_template=student_prompt_template,
    )
    base_teacher_prompt = teacher_template.format(
        gt_route=gt_route,
        student_prompt=student["text"],
        teacher_output_max_tokens=max_output_tokens,
    )
    attempts: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None

    for attempt in range(1, retries + 1):
        teacher_prompt = base_teacher_prompt
        if attempts:
            teacher_prompt += (
                "\n\nThe previous answer was rejected by automatic validation. "
                "Regenerate the complete answer from scratch and correct this issue:\n"
                f"{attempts[-1]['error']}"
            )
        messages = multimodal_messages(
            system="",
            user_text=teacher_prompt,
            image_path=student["image_path"],
        )
        raw = ""
        error = ""
        attempt_started = time.monotonic()
        try:
            raw = chat_text(
                clients.get(),
                model,
                messages,
                temperature=temperature,
                max_tokens=teacher_max_tokens,
            )
            accepted = validate_teacher_output(
                raw,
                gt_route=gt_route,
                tokenizer=tokenizer,
                min_reasoning_tokens=min_reasoning_tokens,
                max_output_tokens=max_output_tokens,
                validation_profile=validation_profile,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        attempts.append(
            {
                "sample_id": rec["sample_id"],
                "attempt": attempt,
                "accepted": accepted is not None,
                "error": error,
                "seconds": round(time.monotonic() - attempt_started, 3),
                "raw": raw,
            }
        )
        if accepted is not None:
            break
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 8))

    common = {
        "sample_id": rec["sample_id"],
        "domain": rec["domain"],
        "gt_route": gt_route,
        "teacher_model": model,
        "teacher_prompt": str(teacher_prompt_path.resolve()),
        "student_prompt": student["text"],
        "image_path": student["image_path"],
        "attempt_count": len(attempts),
        "seconds": round(time.monotonic() - started, 3),
        "attempts": attempts,
    }
    if accepted is None:
        return {
            **common,
            "accepted": False,
            "error": attempts[-1]["error"] if attempts else "no attempt was made",
        }

    image_prefix = "<image>\n" if student["image_path"] else ""
    return {
        **common,
        "accepted": True,
        "reasoning": accepted["reasoning"],
        "reasoning_tokens": accepted["reasoning_tokens"],
        "output_tokens": accepted["output_tokens"],
        "output": accepted["output"],
        "conversations": [
            {"from": "human", "value": image_prefix + student["text"]},
            {"from": "gpt", "value": accepted["output"]},
        ],
        "images": [student["image_path"]] if student["image_path"] else [],
        "response_format": "reasoning_response",
        "validation_profile": validation_profile,
        "skill_markers": accepted["skill_markers"],
        "n_skills": (
            len(normalize_skill_list(skill_bank))
            if skill_bank
            else len(
                set(
                    re.findall(
                        r"(?m)^\s*(\d+)\.\s+\S",
                        student_prompt_template or "",
                    )
                )
            )
        ),
        "skills_block_preview": (
            format_skills_block(skill_bank)[:400] if skill_bank else "inline"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("metromap", "travelmap"))
    parser.add_argument("--skills_json", type=Path)
    parser.add_argument(
        "--student_prompt_path",
        type=Path,
        help="Use this exact student prompt template instead of the canonical prompt.",
    )
    parser.add_argument(
        "--teacher_prompt_path",
        type=Path,
        help="Use this teacher prompt template instead of the domain SFT teacher prompt.",
    )
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--raw_output_file", type=Path)
    parser.add_argument("--failures_file", type=Path)
    parser.add_argument("--selection_file", type=Path)
    parser.add_argument("--prompt_preview", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 selects the full training set")
    parser.add_argument("--sample_seed", type=int, default=20260904)
    parser.add_argument(
        "--stratify_difficulty",
        action="store_true",
        help=(
            "Preserve the eligible source distribution of Map_Difficulty x "
            "Query_Difficulty while sampling."
        ),
    )
    parser.add_argument(
        "--difficulty_quotas",
        help=(
            "Exact Map_Difficulty x Query_Difficulty quotas as a JSON object or "
            "JSON file, for example '{\"HardxHard\": 100, \"HardxMedium\": 50}'."
        ),
    )
    parser.add_argument(
        "--exclude_start_transfer",
        action="store_true",
        help="Exclude samples whose GT route marks station_1 with (transfer).",
    )
    parser.add_argument(
        "--exclude_sample_ids_file",
        type=Path,
        help="Exclude IDs listed in a selection JSON, JSON list, or JSONL file.",
    )
    parser.add_argument(
        "--include_sample_ids_file",
        type=Path,
        help="Restrict generation to IDs listed in a selection JSON, JSON list, or JSONL file.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mode", choices=("teacher",), default="teacher")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--teacher_max_tokens", type=int, default=2048)
    parser.add_argument(
        "--min_reasoning_tokens",
        "--min_think_tokens",
        dest="min_reasoning_tokens",
        type=int,
        default=0,
    )
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument(
        "--tokenizer_path",
        default=str(ROOT / "models/Qwen3.5-9B"),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--validation_profile",
        choices=VALIDATION_PROFILES,
        default="legacy",
        help="Validate either the legacy skill-selection format or D4 inline markers.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip_failed_on_resume",
        action="store_true",
        help="When resuming, treat sample IDs already recorded in the failures file as completed.",
    )
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.min_reasoning_tokens >= args.max_output_tokens:
        raise SystemExit(
            "--min_reasoning_tokens must be smaller than --max_output_tokens"
        )

    skill_bank = (
        json.loads(args.skills_json.read_text(encoding="utf-8"))
        if args.skills_json
        else None
    )
    skills = normalize_skill_list(skill_bank) if skill_bank else []
    if args.skills_json and not skills:
        raise SystemExit(f"No skills found in {args.skills_json}")
    if not args.student_prompt_path and not skills:
        raise SystemExit(
            "Provide --skills_json for the canonical prompt or "
            "--student_prompt_path for a self-contained prompt."
        )

    student_prompt_template = (
        args.student_prompt_path.read_text(encoding="utf-8")
        if args.student_prompt_path
        else None
    )
    include_skills = bool(skills) and (
        student_prompt_template is None
        or "__PLANNING_SKILLS__" in student_prompt_template
    )
    teacher_prompt_path = args.teacher_prompt_path or _teacher_prompt_path(args.domain)
    teacher_template = teacher_prompt_path.read_text(encoding="utf-8")
    all_rows = load_planning(args.domain, "train", only_vertex2=True)
    eligible_rows = (
        [row for row in all_rows if not gt_starts_with_transfer(row)]
        if args.exclude_start_transfer
        else all_rows
    )
    excluded_start_transfer = len(all_rows) - len(eligible_rows)
    included_sample_ids = load_excluded_sample_ids(args.include_sample_ids_file)
    if args.include_sample_ids_file:
        source_ids = {str(row["sample_id"]) for row in all_rows}
        missing_included_ids = included_sample_ids - source_ids
        if missing_included_ids:
            examples = sorted(missing_included_ids)[:5]
            raise SystemExit(
                f"{len(missing_included_ids)} included sample IDs are absent from "
                f"the source split; examples={examples}"
            )
        eligible_rows = [
            row
            for row in eligible_rows
            if str(row["sample_id"]) in included_sample_ids
        ]
    excluded_sample_ids = load_excluded_sample_ids(args.exclude_sample_ids_file)
    before_id_exclusion = len(eligible_rows)
    eligible_rows = [
        row for row in eligible_rows if str(row["sample_id"]) not in excluded_sample_ids
    ]
    excluded_by_sample_id = before_id_exclusion - len(eligible_rows)
    difficulty_quotas = parse_difficulty_quotas(args.difficulty_quotas)
    if difficulty_quotas and args.stratify_difficulty:
        raise SystemExit("Use either --difficulty_quotas or --stratify_difficulty, not both")
    if difficulty_quotas:
        quota_total = sum(difficulty_quotas.values())
        if args.limit not in (0, quota_total):
            raise SystemExit(
                f"--limit must be 0 or match the difficulty quota total ({quota_total})"
            )
        selected = select_difficulty_quotas(
            eligible_rows,
            quotas=difficulty_quotas,
            seed=args.sample_seed,
        )
    else:
        selection_fn = (
            select_difficulty_stratified
            if args.stratify_difficulty
            else select_map_stratified
        )
        selected = selection_fn(eligible_rows, limit=args.limit, seed=args.sample_seed)
    if not selected:
        raise SystemExit("No eligible samples remain after applying selection filters.")
    unique_maps = len({str(row.get("figure") or "") for row in selected})

    selection_file = args.selection_file or args.output_file.with_suffix(".selection.json")
    selection = {
        "domain": args.domain,
        "source_count": len(all_rows),
        "eligible_count": len(eligible_rows),
        "excluded_start_transfer": excluded_start_transfer,
        "excluded_by_sample_id": excluded_by_sample_id,
        "exclude_sample_ids_file": (
            str(args.exclude_sample_ids_file.resolve())
            if args.exclude_sample_ids_file
            else None
        ),
        "include_sample_ids_file": (
            str(args.include_sample_ids_file.resolve())
            if args.include_sample_ids_file
            else None
        ),
        "included_sample_ids": len(included_sample_ids),
        "selected_count": len(selected),
        "unique_maps": unique_maps,
        "sample_seed": args.sample_seed,
        "difficulty_stratified": args.stratify_difficulty,
        "requested_difficulty_quotas": {
            f"{map_level}x{query_level}": count
            for (map_level, query_level), count in sorted(difficulty_quotas.items())
        },
        "difficulty_counts": {
            f"{map_level}x{query_level}": sum(
                1
                for row in selected
                if _difficulty_key(row) == (map_level, query_level)
            )
            for map_level, query_level in sorted({_difficulty_key(row) for row in selected})
        },
        "sample_ids": [row["sample_id"] for row in selected],
    }
    first_bundle = build_prompt_bundle(
        selected[0],
        skill_bank=skill_bank,
        include_skills=include_skills,
        thinking=False,
        prompt_template=student_prompt_template,
    )
    first_teacher_prompt = teacher_template.format(
        gt_route=selected[0]["gt_route"],
        student_prompt=first_bundle["text"],
        teacher_output_max_tokens=args.max_output_tokens,
    )
    print(
        f"[{args.domain}] source={len(all_rows)} eligible={len(eligible_rows)} "
        f"excluded_start_transfer={excluded_start_transfer} selected={len(selected)} "
        f"excluded_by_sample_id={excluded_by_sample_id} "
        f"unique_maps={unique_maps} workers={args.workers} model={args.model}",
        flush=True,
    )
    if args.prepare_only:
        _write_json(selection_file, selection)
        print("Prepared selection and prompt preview; no API request was sent.")
        return

    raw_output = args.raw_output_file or args.output_file.with_suffix(".raw.jsonl")
    failures_output = args.failures_file or args.output_file.with_suffix(".failures.jsonl")
    generation_manifest = args.output_file.with_suffix(".manifest.json")
    manifest_payload = {
        "schema_version": 1,
        "domain": args.domain,
        "model": args.model,
        "student_prompt_path": (
            str(args.student_prompt_path.resolve()) if args.student_prompt_path else None
        ),
        "student_prompt": student_prompt_template,
        "skills_json_path": str(args.skills_json.resolve()) if args.skills_json else None,
        "skill_bank": skill_bank,
        "include_skills": include_skills,
        "teacher_prompt_path": str(teacher_prompt_path.resolve()),
        "teacher_prompt": teacher_template,
        "selection": selection,
        "temperature": args.temperature,
        "teacher_max_tokens": args.teacher_max_tokens,
        "min_reasoning_tokens": args.min_reasoning_tokens,
        "max_output_tokens": args.max_output_tokens,
        "validation_profile": args.validation_profile,
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
    }
    lock_generation_manifest(
        generation_manifest,
        manifest_payload,
        artifacts=(args.output_file, raw_output, failures_output),
        resume=args.resume,
    )
    _write_json(selection_file, selection)
    if args.prompt_preview:
        args.prompt_preview.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_preview.write_text(first_teacher_prompt, encoding="utf-8")

    tokenizer = load_token_counter(args.tokenizer_path)
    # Fail before opening output files if the key or SDK is unavailable.
    openlux_client(timeout=args.timeout)
    clients = ThreadClients(timeout=args.timeout)

    completed_ids = {
        str(row.get("sample_id"))
        for row in (_read_jsonl(args.output_file) if args.resume else [])
        if row.get("accepted", True)
    }
    skipped_failure_ids = set()
    if args.resume and args.skip_failed_on_resume:
        skipped_failure_ids = {
            str(row.get("sample_id"))
            for row in _read_jsonl(failures_output)
            if row.get("sample_id")
        }
        completed_ids.update(skipped_failure_ids)
    pending = [row for row in selected if row["sample_id"] not in completed_ids]
    print(
        f"resume={args.resume} completed={len(completed_ids)} "
        f"skipped_failures={len(skipped_failure_ids)} pending={len(pending)}",
        flush=True,
    )

    for path in (args.output_file, raw_output, failures_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume and args.output_file.exists() else "w"
    raw_mode = "a" if args.resume and raw_output.exists() else "w"
    failure_mode = "a" if args.resume and failures_output.exists() else "w"

    success_count = 0
    failure_count = 0
    with (
        args.output_file.open(output_mode, encoding="utf-8") as output_handle,
        raw_output.open(raw_mode, encoding="utf-8") as raw_handle,
        failures_output.open(failure_mode, encoding="utf-8") as failure_handle,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        futures = {
            executor.submit(
                generate_one,
                row,
                skill_bank=skill_bank,
                student_prompt_template=student_prompt_template,
                include_skills=include_skills,
                teacher_template=teacher_template,
                teacher_prompt_path=teacher_prompt_path,
                clients=clients,
                tokenizer=tokenizer,
                model=args.model,
                temperature=args.temperature,
                teacher_max_tokens=args.teacher_max_tokens,
                min_reasoning_tokens=args.min_reasoning_tokens,
                max_output_tokens=args.max_output_tokens,
                retries=args.retries,
                validation_profile=args.validation_profile,
            ): row
            for row in pending
        }
        for position, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "sample_id": row["sample_id"],
                    "domain": args.domain,
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "attempts": [],
                }

            for attempt in result.pop("attempts", []):
                raw_handle.write(json.dumps(attempt, ensure_ascii=False) + "\n")
            raw_handle.flush()

            if result["accepted"]:
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
                success_count += 1
                status = (
                    f"OK reasoning={result['reasoning_tokens']} "
                    f"total={result['output_tokens']} "
                    f"attempts={result['attempt_count']}"
                )
            else:
                failure_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                failure_handle.flush()
                failure_count += 1
                status = f"FAIL {result.get('error', 'unknown error')}"
            print(f"[{position}/{len(pending)}] {result['sample_id']} {status}", flush=True)

    total_complete = len(completed_ids) + success_count
    print(
        f"finished accepted={total_complete}/{len(selected)} "
        f"new_success={success_count} new_failure={failure_count}",
        flush=True,
    )
    print(f"labels: {args.output_file}")
    print(f"raw attempts: {raw_output}")
    print(f"failures: {failures_output}")
    if failure_count:
        raise SystemExit("Some labels failed validation; rerun the same command with --resume.")


if __name__ == "__main__":
    main()
