#!/usr/bin/env python3
"""Rewrite each evolved candidate's usage guidance and few-shots for its new Skill Bank."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


from rlharness.common.api_client import extract_json_obj, openlux_client
from rlharness.common.config import ROOT


DEFAULT_TEMPLATE = ROOT / "prompts/evolution/metromap/rewrite.txt"
MARKER_RE = re.compile(r"\[Using Skill\s+(\d+)\]", re.IGNORECASE)
SKILL_SELECTION_RE = re.compile(r"\bSkill\s+(\d+)\b", re.IGNORECASE)
RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
EXAMPLE_RE = re.compile(r"(?=^## Example\s+\d+)", re.MULTILINE)
DECIMAL_RE = re.compile(r"(?<![\w.])\d+\.\d+(?!\w|\.\d)")
ROUTE_LINE_RE = re.compile(r"^[A-Z]:\s+.+$", re.MULTILINE)
TASK_LINE_RE = re.compile(r"^Task:\s+.+$", re.MULTILINE)


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
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_body(text: str, heading: str, next_heading: str) -> str:
    marker = f"# {heading}\n"
    next_marker = f"\n# {next_heading}\n"
    try:
        return text.split(marker, 1)[1].split(next_marker, 1)[0].strip()
    except IndexError as exc:
        raise ValueError(f"Cannot extract section {heading!r}") from exc


def replace_body(text: str, heading: str, next_heading: str, body: str) -> str:
    marker = f"# {heading}\n"
    next_marker = f"\n# {next_heading}\n"
    before, remainder = text.split(marker, 1)
    _, after = remainder.split(next_marker, 1)
    return before + marker + body.strip() + next_marker + after


def fill_template(template: str, **values: str) -> str:
    rendered = template
    for name, value in values.items():
        marker = "{" + name + "}"
        if marker not in rendered:
            raise ValueError(f"Template is missing {marker}")
        rendered = rendered.replace(marker, value)
    return rendered


def example_blocks(few_shots: str) -> list[str]:
    starts = list(EXAMPLE_RE.finditer(few_shots))
    if not starts:
        return []
    blocks: list[str] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(few_shots)
        blocks.append(few_shots[match.start() : end].strip())
    return blocks


def selection_ids(reasoning: str) -> list[int]:
    match = re.search(
        r"\[Relevant Strategy Selection\](.*?)(?=\n\[[^\]]+\]|\Z)",
        reasoning,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return []
    return [int(value) for value in SKILL_SELECTION_RE.findall(match.group(1))]


def normalized_responses(text: str) -> list[str]:
    return [match.strip() for match in RESPONSE_RE.findall(text)]


def validate_rewrite(
    value: Any,
    *,
    original_few_shots: str,
    skill_count: int,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["API output is not a JSON object"]
    how_to_use = value.get("how_to_use")
    few_shots = value.get("few_shot_demonstrations")
    if not isinstance(how_to_use, str) or not how_to_use.strip():
        errors.append("how_to_use is missing or empty")
    if not isinstance(few_shots, str) or not few_shots.strip():
        errors.append("few_shot_demonstrations is missing or empty")
    if not isinstance(value.get("marker_audit"), list):
        errors.append("marker_audit must be a list")
    if not isinstance(value.get("demonstration_coverage"), list):
        errors.append("demonstration_coverage must be a list")
    if not isinstance(value.get("consistency_warnings"), list):
        errors.append("consistency_warnings must be a list")
    if errors:
        return errors

    how_to_use = how_to_use.strip()
    few_shots = few_shots.strip()
    if how_to_use.startswith("#"):
        errors.append("how_to_use must not include its Markdown heading")
    if few_shots.startswith("# Few-Shot Demonstrations"):
        errors.append("few_shot_demonstrations must not include its top-level heading")
    if "[Relevant Strategy Selection]" not in how_to_use:
        errors.append("how_to_use does not require Relevant Strategy Selection")
    if "[Using Skill N]" not in how_to_use:
        errors.append("how_to_use does not explain the inline Using Skill marker")

    original_examples = example_blocks(original_few_shots)
    rewritten_examples = example_blocks(few_shots)
    if len(rewritten_examples) != len(original_examples):
        errors.append(
            f"few-shot example count changed: {len(original_examples)} -> {len(rewritten_examples)}"
        )

    if normalized_responses(few_shots) != normalized_responses(original_few_shots):
        errors.append("one or more final <response> routes changed")
    if TASK_LINE_RE.findall(few_shots) != TASK_LINE_RE.findall(original_few_shots):
        errors.append("one or more few-shot Task lines changed")
    if ROUTE_LINE_RE.findall(few_shots) != ROUTE_LINE_RE.findall(original_few_shots):
        errors.append("one or more labeled candidate-route lines changed")
    if Counter(DECIMAL_RE.findall(few_shots)) != Counter(DECIMAL_RE.findall(original_few_shots)):
        errors.append("decimal values or their multiplicities changed")

    actual_usage: dict[str, list[int]] = {}
    for index, block in enumerate(rewritten_examples, 1):
        reasoning_match = REASONING_RE.search(block)
        if reasoning_match is None:
            errors.append(f"example {index}: missing <reasoning> block")
            continue
        reasoning = reasoning_match.group(1)
        selected = selection_ids(reasoning)
        used = [int(value) for value in MARKER_RE.findall(reasoning)]
        actual_usage[f"example_{index}"] = used
        all_block_markers = [int(value) for value in MARKER_RE.findall(block)]
        if used != all_block_markers:
            errors.append(f"example {index}: a Using Skill marker appears outside <reasoning>")
        if not selected:
            errors.append(f"example {index}: no skills selected")
        if not used:
            errors.append(f"example {index}: no skills used")
        invalid = sorted({number for number in [*selected, *used] if not 1 <= number <= skill_count})
        if invalid:
            errors.append(f"example {index}: invalid new skill IDs {invalid}")
        duplicated_selection = sorted(number for number, count in Counter(selected).items() if count > 1)
        duplicated_usage = sorted(number for number, count in Counter(used).items() if count > 1)
        if duplicated_selection:
            errors.append(f"example {index}: repeatedly selects skills {duplicated_selection}")
        if duplicated_usage:
            errors.append(f"example {index}: repeatedly marks skills {duplicated_usage}")
        if set(selected) != set(used):
            errors.append(
                f"example {index}: selected skills {sorted(set(selected))} "
                f"do not equal used skills {sorted(set(used))}"
            )
        marker_runs = re.findall(
            r"(?:\[Using Skill\s+\d+\]\s*)+(?P<action>[^\s\[].*)",
            reasoning,
            flags=re.IGNORECASE,
        )
        if used and not marker_runs:
            errors.append(f"example {index}: markers are not followed by an executable action")

    marker_audit = value["marker_audit"]
    audited_examples: set[str] = set()
    for item in marker_audit:
        if not isinstance(item, dict):
            errors.append("marker_audit contains a non-object entry")
            continue
        example_id = str(item.get("example_id") or "")
        if example_id not in actual_usage or example_id in audited_examples:
            errors.append(f"marker_audit has an invalid or duplicate example_id: {example_id!r}")
            continue
        audited_examples.add(example_id)
        try:
            used = [int(number) for number in item.get("used_skill_numbers", [])]
            placements = item.get("placements", [])
            placed = [int(entry.get("skill_number")) for entry in placements]
        except (AttributeError, TypeError, ValueError):
            errors.append(f"{example_id}: malformed marker audit")
            continue
        if len(used) != len(set(used)):
            errors.append(f"{example_id}: marker audit repeats a used skill number")
        if sorted(used) != sorted(actual_usage[example_id]):
            errors.append(f"{example_id}: marker audit does not match rendered markers")
        if sorted(placed) != sorted(used):
            errors.append(f"{example_id}: placement audit does not match used skills")
        if any(not str(entry.get("following_action") or "").strip() for entry in placements):
            errors.append(f"{example_id}: a marker placement has no following action")
    if audited_examples != set(actual_usage):
        errors.append("marker_audit does not account for every demonstration exactly once")

    coverage = value["demonstration_coverage"]
    coverage_ids: list[int] = []
    examples_by_skill = {
        skill: sorted(example_id for example_id, used in actual_usage.items() if skill in used)
        for skill in range(1, skill_count + 1)
    }
    for item in coverage:
        if not isinstance(item, dict):
            errors.append("demonstration_coverage contains a non-object entry")
            continue
        try:
            skill_number = int(item.get("skill_number"))
        except (TypeError, ValueError):
            errors.append("demonstration_coverage contains an invalid skill_number")
            continue
        coverage_ids.append(skill_number)
        illustrated = sorted(str(value) for value in item.get("illustrated_in_examples", []))
        if illustrated != examples_by_skill.get(skill_number, []):
            errors.append(f"skill {skill_number}: demonstration coverage does not match markers")
        expected_status = "NATURALLY_ILLUSTRATED" if illustrated else "UNILLUSTRATED"
        if str(item.get("status") or "") != expected_status:
            errors.append(f"skill {skill_number}: expected coverage status {expected_status}")
    if sorted(coverage_ids) != list(range(1, skill_count + 1)):
        errors.append("demonstration_coverage must account for every new skill exactly once")
    return errors


def materialize_rewrite(
    *,
    candidate_dir: Path,
    source_prompt: str,
    original_few_shots: str,
    skill_count: int,
    model: str,
    value: dict[str, Any],
    raw: str,
    attempt: int,
    attempt_errors: list[dict[str, Any]],
    provenance: dict[str, str],
) -> str:
    pre_rewrite = (candidate_dir / "pre_rewrite_prompt.txt").read_text(encoding="utf-8")
    final_prompt = replace_body(
        pre_rewrite,
        "How to Use the Skills",
        "Required Procedure",
        value["how_to_use"],
    )
    final_prompt = replace_body(
        final_prompt,
        "Few-Shot Demonstrations",
        "Output Format Constraints",
        value["few_shot_demonstrations"],
    )
    required_procedure_preserved = (
        extract_body(final_prompt, "Required Procedure", "Few-Shot Demonstrations")
        == extract_body(source_prompt, "Required Procedure", "Few-Shot Demonstrations")
    )
    if not required_procedure_preserved:
        raise RuntimeError("Required Procedure changed during deterministic materialization")

    (candidate_dir / "full_prompt.txt").write_text(final_prompt, encoding="utf-8")
    (candidate_dir / "fewshot_rewrite.raw.txt").write_text(raw + "\n", encoding="utf-8")
    (candidate_dir / "fewshot_rewrite.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "valid": True,
        "model": model,
        "attempt": attempt,
        "skill_count": skill_count,
        "example_count": len(example_blocks(original_few_shots)),
        "demonstration_coverage": value["demonstration_coverage"],
        "consistency_warnings": value["consistency_warnings"],
        "required_procedure_preserved": required_procedure_preserved,
        "previous_attempt_errors": attempt_errors,
        **provenance,
    }
    (candidate_dir / "fewshot_rewrite.validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return f"{candidate_dir.name}: few-shots rewritten (attempt {attempt})"


def rewrite_one(
    *,
    candidate_dir: Path,
    source_prompt: str,
    template: str,
    clients: ThreadClients,
    model: str,
    timeout_retries: int,
    dry_run: bool,
) -> str:
    candidate_path = candidate_dir / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    skills = (candidate_dir / "planning_skills.txt").read_text(encoding="utf-8").strip()
    skill_count = len(candidate["skills"])
    original_how_to_use = extract_body(source_prompt, "How to Use the Skills", "Required Procedure")
    original_few_shots = extract_body(
        source_prompt, "Few-Shot Demonstrations", "Output Format Constraints"
    )
    request = fill_template(
        template,
        new_skill_bank=skills,
        operation_log=json.dumps(candidate.get("operation_log", []), ensure_ascii=False, indent=2),
        original_how_to_use=original_how_to_use,
        original_few_shots=original_few_shots,
    )
    provenance = {
        "model": model,
    }
    validation_path = candidate_dir / "fewshot_rewrite.validation.json"
    final_path = candidate_dir / "full_prompt.txt"
    request_path = candidate_dir / "fewshot_rewrite.request.txt"
    rewrite_artifacts = (
        validation_path,
        final_path,
        candidate_dir / "fewshot_rewrite.json",
        candidate_dir / "fewshot_rewrite.raw.txt",
    )
    if request_path.is_file():
        if request_path.read_text(encoding="utf-8") != request:
            raise RuntimeError(
                f"{candidate_dir.name}: rewrite request changed; use a new evolution run directory"
            )
    elif any(path.exists() for path in rewrite_artifacts):
        raise RuntimeError(
            f"{candidate_dir.name}: existing rewrite has no request snapshot; "
            "use a new evolution run directory"
        )
    else:
        request_path.write_text(request, encoding="utf-8")
    if validation_path.is_file() and final_path.is_file():
        existing_validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if existing_validation.get("valid") is True and all(
            existing_validation.get(key) == value for key, value in provenance.items()
        ):
            return f"{candidate_dir.name}: cached validated rewrite"
    if dry_run:
        return f"{candidate_dir.name}: rewrite request prepared"

    attempt_errors: list[dict[str, Any]] = []
    saved_attempts: list[tuple[int, Path]] = []
    for raw_path in candidate_dir.glob("fewshot_rewrite.raw_attempt_*.txt"):
        match = re.search(r"_(\d+)\.txt$", raw_path.name)
        if match:
            saved_attempts.append((int(match.group(1)), raw_path))
    for attempt, raw_path in sorted(saved_attempts, reverse=True):
        try:
            raw = raw_path.read_text(encoding="utf-8").strip()
            value = extract_json_obj(raw)
            errors = validate_rewrite(
                value,
                original_few_shots=original_few_shots,
                skill_count=skill_count,
            )
            if not errors:
                return materialize_rewrite(
                    candidate_dir=candidate_dir,
                    source_prompt=source_prompt,
                    original_few_shots=original_few_shots,
                    skill_count=skill_count,
                    model=model,
                    value=value,
                    raw=raw,
                    attempt=attempt,
                    attempt_errors=attempt_errors,
                    provenance=provenance,
                ) + " [recovered]"
            attempt_errors.append({"attempt": attempt, "errors": errors, "source": "saved"})
        except Exception as exc:
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "source": "saved",
                }
            )

    for attempt in range(1, timeout_retries + 1):
        retry_feedback = ""
        if attempt_errors:
            cumulative_errors = list(
                dict.fromkeys(
                    error
                    for record in attempt_errors
                    for error in record["errors"]
                )
            )
            retry_feedback = (
                "\n\n# Cumulative Validation Errors From Previous Attempts\n"
                + "\n".join(f"- {item}" for item in cumulative_errors)
                + "\nReturn a corrected complete JSON object."
            )
        try:
            response = clients.get().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request + retry_feedback}],
                temperature=0.2,
                max_tokens=12000,
            )
            raw = (response.choices[0].message.content or "").strip()
            (candidate_dir / f"fewshot_rewrite.raw_attempt_{attempt}.txt").write_text(
                raw + "\n", encoding="utf-8"
            )
            value = extract_json_obj(raw)
            errors = validate_rewrite(
                value,
                original_few_shots=original_few_shots,
                skill_count=skill_count,
            )
            if errors:
                attempt_errors.append({"attempt": attempt, "errors": errors})
                continue
            return materialize_rewrite(
                candidate_dir=candidate_dir,
                source_prompt=source_prompt,
                original_few_shots=original_few_shots,
                skill_count=skill_count,
                model=model,
                value=value,
                raw=raw,
                attempt=attempt,
                attempt_errors=attempt_errors,
                provenance=provenance,
            )
        except Exception as exc:
            attempt_errors.append(
                {"attempt": attempt, "errors": [f"{type(exc).__name__}: {exc}"]}
            )
        if attempt < timeout_retries:
            time.sleep(min(2 ** (attempt - 1) + random.random(), 10))

    validation_path.write_text(
        json.dumps(
            {"valid": False, "model": model, "attempt_errors": attempt_errors},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (candidate_dir / "full_prompt.txt").unlink(missing_ok=True)
    raise RuntimeError(f"few-shot rewrite failed after {timeout_retries} attempts")


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 1:
        raise SystemExit("--workers and --retries must be positive")
    manifest = json.loads((args.candidate_root / "manifest.json").read_text(encoding="utf-8"))
    names = [str(name) for name in manifest.get("candidate_names", [])]
    if not names:
        raise SystemExit("Candidate manifest contains no candidate_names")
    source_prompt = (args.candidate_root / "source_prompt.txt").read_text(encoding="utf-8")
    template = args.template.read_text(encoding="utf-8")
    clients = ThreadClients(args.timeout)

    failed = False
    with ThreadPoolExecutor(max_workers=min(args.workers, len(names))) as executor:
        futures = {
            executor.submit(
                rewrite_one,
                candidate_dir=args.candidate_root / name,
                source_prompt=source_prompt,
                template=template,
                clients=clients,
                model=args.model,
                timeout_retries=args.retries,
                dry_run=args.dry_run,
            ): name
            for name in names
        }
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failed = True
                print(f"{futures[future]}: FAILED: {exc}", file=sys.stderr, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
