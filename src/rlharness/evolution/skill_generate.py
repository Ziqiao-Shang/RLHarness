#!/usr/bin/env python3
"""Generate three complete Skill Bank candidates from reusable rollout evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


from rlharness.common.api_client import extract_json_obj, openlux_client
from rlharness.common.config import ROOT

DEFAULT_S0 = ROOT / "prompts/student/metromap/original.txt"
DEFAULT_MINED_EVIDENCE = ROOT / "artifacts/skill_evolution/evidence128"
DEFAULT_TEMPLATE = ROOT / "prompts/evolution/metromap/generate.txt"
DEFAULT_OUTPUT = ROOT / "artifacts/skill_evolution/candidates"

STYLES = {
    "candidate_a_conservative": (
        "Conservative rewrite: preserve the pruned bank's semantic coverage and "
        "make only strongly evidenced merges or clarifications. Minimize behavioral blast radius."
    ),
    "candidate_b_stage_based": (
        "Stage-based rewrite: reorganize overlapping skills into the normal forward planning "
        "stages while retaining all evidence-backed safeguards."
    ),
    "candidate_c_compact": (
        "Compact rewrite: use the fewest independently executable skills that still cover "
        "topology, candidate coverage, route-to-table alignment, task-specific scoring, "
        "comparison, and verification."
    ),
}


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
    parser.add_argument("--s0", type=Path, default=DEFAULT_S0)
    parser.add_argument(
        "--mined-evidence",
        type=Path,
        default=DEFAULT_MINED_EVIDENCE,
        help="Output directory produced by skill_evidence prepare/mine.",
    )
    parser.add_argument("--expected-evidence-count", type=int, default=128)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_skills(prompt: str) -> str:
    try:
        return prompt.split("# Planning Skills\n", 1)[1].split("\n# How to Use the Skills\n", 1)[0].strip()
    except IndexError as exc:
        raise ValueError("S0 does not contain the expected Planning Skills section") from exc


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_mined_evidence(root: Path) -> dict[str, Any]:
    selection = read_json(root / "selection.json")
    analysis_manifest_path = root / "analysis_manifest.json"
    analysis_manifest = read_json(analysis_manifest_path) if analysis_manifest_path.is_file() else {}
    report_paths = sorted((root / "teacher_analysis").glob("batch_*.json"))
    if not report_paths:
        raise FileNotFoundError(f"No teacher analysis reports found under {root}")
    expected_batch_count = int(selection.get("batch_count") or 0)
    if len(report_paths) != expected_batch_count:
        raise ValueError(
            f"Analysis report count mismatch: expected {expected_batch_count}, "
            f"found {len(report_paths)}"
        )
    reports = [read_json(path) for path in report_paths]
    selected_ids = [str(case_id) for case_id in selection.get("sample_ids", [])]
    analyzed_id_list = [
        str(case_id)
        for report in reports
        for case_id in report.get("trajectory_ids", [])
    ]
    if len(analyzed_id_list) != len(set(analyzed_id_list)):
        raise ValueError("Teacher analysis reports contain duplicate trajectory IDs")
    if set(analyzed_id_list) != set(selected_ids):
        raise ValueError("Teacher analysis reports do not exactly cover selection.json")
    analyzed_ids = sorted(analyzed_id_list)
    aggregate: dict[str, list[Any]] = {
        key: []
        for key in (
            "successful_strategies",
            "failure_patterns",
            "existing_skill_findings",
            "missing_strategies",
            "anti_patterns",
        )
    }
    for report in reports:
        batch_id = report.get("batch_id")
        for key in aggregate:
            for item in report.get(key, []):
                if isinstance(item, dict):
                    item = {**item, "source_batch": batch_id}
                aggregate[key].append(item)
    return {
        "source_note": (
            f"{analysis_manifest.get('analysis_model', 'Teacher-model')} batch analyses "
            "of a fixed post-RL student rollout sample. "
            "These reports are diagnostic evidence, not edits that must be accepted."
        ),
        "analysis_model": analysis_manifest.get("analysis_model"),
        "analysis_prompt": analysis_manifest.get("analysis_prompt"),
        "sample_count": int(selection.get("sample_count") or len(analyzed_ids)),
        "batch_count": len(reports),
        "analyzed_trajectory_count": len(analyzed_ids),
        "correctness": selection.get("correctness", {}),
        "difficulty": selection.get("difficulty", {}),
        "aggregate_reports": aggregate,
    }


def render(template: str, **values: str) -> str:
    result = template
    for key, value in values.items():
        marker = "{" + key + "}"
        if marker not in result:
            raise ValueError(f"Template is missing {marker}")
        result = result.replace(marker, value)
    return result


def candidate_validation_errors(
    value: Any,
    name: str,
    source_skill_ids: set[int],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict) or value.get("candidate_name") != name:
        return [f"candidate_name must be exactly {name!r}"]
    skills = value.get("skills")
    if not isinstance(skills, list) or not skills:
        return ["skills must be a non-empty list"]
    required = ("title", "trigger", "action", "guard", "check")
    for index, skill in enumerate(skills, 1):
        if not isinstance(skill, dict) or any(
            not str(skill.get(key, "")).strip() for key in required
        ):
            errors.append(
                f"skill {index} must contain non-empty title/trigger/action/guard/check"
            )
    if errors:
        return errors
    titles = {str(skill["title"]).strip().casefold() for skill in skills}
    if len(titles) != len(skills):
        errors.append("skill titles must be unique")

    operation_log = value.get("operation_log")
    if not isinstance(operation_log, list):
        return [*errors, "operation_log must be a list"]
    covered_by_operations: list[int] = []
    operation_by_source: dict[int, str] = {}
    operation_results_by_source: dict[int, list[str]] = {}
    for index, item in enumerate(operation_log, 1):
        if not isinstance(item, dict):
            errors.append(f"operation_log entry {index} must be an object")
            continue
        operation = str(item.get("operation") or "").upper()
        if operation not in {"KEEP", "REWRITE", "MERGE", "INSERT", "DELETE"}:
            errors.append(f"operation_log entry {index} has invalid operation {operation!r}")
            continue
        try:
            sources = [int(number) for number in item.get("source_skill_numbers", [])]
        except (TypeError, ValueError):
            errors.append(f"operation_log entry {index} has invalid source_skill_numbers")
            continue
        result_titles = [str(title).strip() for title in item.get("result_titles", [])]
        if not str(item.get("evidence") or "").strip():
            errors.append(f"operation_log entry {index} must provide evidence")
        if any(number not in source_skill_ids for number in sources):
            errors.append(f"operation_log entry {index} refers to an unknown source skill")
        if operation == "INSERT":
            if sources or not result_titles:
                errors.append(
                    f"INSERT entry {index} must have no sources and at least one result title"
                )
        elif not sources:
            errors.append(f"{operation} entry {index} must name at least one source skill")
        if operation == "DELETE":
            if result_titles:
                errors.append(f"DELETE entry {index} must not have result titles")
        elif not result_titles or any(
            title.casefold() not in titles for title in result_titles
        ):
            errors.append(
                f"operation_log entry {index} contains a missing or unknown result title"
            )
        for source in sources:
            if source in operation_by_source:
                errors.append(f"source skill {source} appears in more than one operation")
            operation_by_source[source] = operation
            operation_results_by_source[source] = result_titles
        covered_by_operations.extend(sources)
    missing_operations = sorted(source_skill_ids - set(covered_by_operations))
    if missing_operations:
        errors.append(f"operation_log omits source skills {missing_operations}")

    coverage_audit = value.get("coverage_audit")
    if not isinstance(coverage_audit, list):
        return [*errors, "coverage_audit must be a list"]
    disposition_for_operation = {
        "KEEP": "PRESERVED",
        "REWRITE": "REWRITTEN",
        "MERGE": "MERGED",
        "DELETE": "DELETED",
    }
    audited_sources: list[int] = []
    for index, item in enumerate(coverage_audit, 1):
        if not isinstance(item, dict):
            errors.append(f"coverage_audit entry {index} must be an object")
            continue
        try:
            source = int(item.get("source_skill_number"))
        except (TypeError, ValueError):
            errors.append(f"coverage_audit entry {index} has an invalid source skill")
            continue
        responsibilities = item.get("core_responsibilities")
        covered_titles = [
            str(title).strip() for title in item.get("covered_by_result_titles", [])
        ]
        deletion_evidence = item.get("deletion_evidence")
        if (
            source not in source_skill_ids
            or not isinstance(responsibilities, list)
            or not responsibilities
        ):
            errors.append(
                f"coverage_audit entry {index} has an unknown source or empty responsibilities"
            )
            continue
        if any(not str(responsibility).strip() for responsibility in responsibilities):
            errors.append(f"coverage_audit entry {index} contains an empty responsibility")
        operation = operation_by_source.get(source)
        if operation is None:
            errors.append(f"coverage_audit source skill {source} has no operation_log entry")
            continue
        expected_disposition = disposition_for_operation[operation]
        if str(item.get("disposition") or "").upper() != expected_disposition:
            errors.append(
                f"source skill {source} disposition must be {expected_disposition} "
                f"because its operation is {operation}"
            )
        if not isinstance(deletion_evidence, list):
            errors.append(f"source skill {source} deletion_evidence must be a list")
            continue
        if expected_disposition == "DELETED":
            if covered_titles or not deletion_evidence:
                errors.append(
                    f"deleted source skill {source} needs deletion evidence and no result titles"
                )
        else:
            if not covered_titles or any(
                title.casefold() not in titles for title in covered_titles
            ):
                errors.append(
                    f"source skill {source} coverage contains a missing or unknown title"
                )
            operation_titles = {
                title.casefold() for title in operation_results_by_source[source]
            }
            audit_titles = {title.casefold() for title in covered_titles}
            if not operation_titles.issubset(audit_titles):
                errors.append(
                    f"source skill {source} coverage must include every result title "
                    "from its operation"
                )
            if deletion_evidence:
                errors.append(
                    f"non-deleted source skill {source} must have empty deletion_evidence"
                )
        audited_sources.append(source)
    duplicated_audits = sorted(
        source for source, count in Counter(audited_sources).items() if count > 1
    )
    if duplicated_audits:
        errors.append(f"coverage_audit repeats source skills {duplicated_audits}")
    missing_audits = sorted(source_skill_ids - set(audited_sources))
    if missing_audits:
        errors.append(f"coverage_audit omits source skills {missing_audits}")
    return errors


def valid_candidate(value: Any, name: str, source_skill_ids: set[int]) -> bool:
    return not candidate_validation_errors(value, name, source_skill_ids)


def candidate_text(value: dict[str, Any]) -> str:
    lines = ["# Planning Skills", ""]
    for index, skill in enumerate(value["skills"], 1):
        lines.extend(
            [
                f"{index}. {skill['title'].strip()}",
                f"   Trigger: {skill['trigger'].strip()}",
                f"   Action: {skill['action'].strip()}",
                f"   Guard: {skill['guard'].strip()}",
                f"   Check: {skill['check'].strip()}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def materialize_full_prompt(source_prompt: str, skills: str) -> str:
    """Replace only the Skill Bank; a later API stage adapts usage guidance and few-shots."""
    prefix, remainder = source_prompt.split("# Planning Skills\n", 1)
    _, suffix = remainder.split("# How to Use the Skills\n", 1)
    return prefix + skills.rstrip() + "\n\n# How to Use the Skills\n" + suffix


def lock_text_snapshot(path: Path, expected: str, *, label: str, require_existing: bool) -> None:
    """Create an exact text snapshot or reject conflicting run contents."""
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"{label} differs from the existing run: {path}. Use a new --output.")
        return
    if require_existing:
        raise SystemExit(f"Existing candidate run has no {label}: {path}. Use a new --output.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(expected, encoding="utf-8")
    temporary.replace(path)


def candidate_is_complete(
    candidate_dir: Path,
    *,
    name: str,
    model: str,
    source_skill_ids: set[int],
    source_prompt: str,
) -> bool:
    """Return true only for a complete, internally consistent generated candidate."""
    candidate_path = candidate_dir / "candidate.json"
    skills_path = candidate_dir / "planning_skills.txt"
    pre_rewrite_path = candidate_dir / "pre_rewrite_prompt.txt"
    if not all(path.is_file() for path in (candidate_path, skills_path, pre_rewrite_path)):
        return False
    try:
        value = read_json(candidate_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if candidate_validation_errors(value, name, source_skill_ids):
        return False
    if value.get("generation", {}).get("model") != model:
        return False
    expected_skills = candidate_text(value)
    if skills_path.read_text(encoding="utf-8") != expected_skills:
        return False
    expected_prompt = materialize_full_prompt(source_prompt, expected_skills)
    return pre_rewrite_path.read_text(encoding="utf-8") == expected_prompt


def call_one(
    *, clients: ThreadClients, model: str, prompt: str, name: str,
    retries: int, output: Path, source_skill_ids: set[int],
) -> str:
    candidate_dir = output / name
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            retry_feedback = ""
            if errors:
                retry_feedback = (
                    "\n\n# Deterministic Validation Errors From Previous Attempts\n"
                    + "\n".join(f"- {error}" for error in errors[-20:])
                    + "\nReturn a corrected complete JSON object."
                )
            response = clients.get().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt + retry_feedback}],
                temperature=0.35,
                max_tokens=8192,
            )
            raw = (response.choices[0].message.content or "").strip()
            (candidate_dir / f"raw_attempt_{attempt}.txt").write_text(
                raw + "\n", encoding="utf-8"
            )
            value = extract_json_obj(raw)
            validation_errors = candidate_validation_errors(value, name, source_skill_ids)
            if validation_errors:
                errors.extend(
                    f"attempt {attempt}: {error}" for error in validation_errors
                )
                continue
            value["generation"] = {"model": model, "attempt": attempt}
            (candidate_dir / "raw_response.txt").write_text(raw + "\n", encoding="utf-8")
            (candidate_dir / "candidate.json").write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            skills = candidate_text(value)
            (candidate_dir / "planning_skills.txt").write_text(skills, encoding="utf-8")
            source_prompt = (output / "source_prompt.txt").read_text(encoding="utf-8")
            (candidate_dir / "pre_rewrite_prompt.txt").write_text(
                materialize_full_prompt(source_prompt, skills), encoding="utf-8"
            )
            # Never let validation consume a stale final prompt from an older run.
            for stale in (
                "full_prompt.txt",
                "fewshot_rewrite.json",
                "fewshot_rewrite.raw.txt",
                "fewshot_rewrite.request.txt",
                "fewshot_rewrite.validation.json",
            ):
                (candidate_dir / stale).unlink(missing_ok=True)
            (candidate_dir / "errors.json").unlink(missing_ok=True)
            return f"{name}: complete ({len(value['skills'])} skills)"
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
    (candidate_dir / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raise RuntimeError(f"{name} failed after {retries} attempts")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    skill_bank = extract_skills(args.s0.read_text(encoding="utf-8"))
    source_skill_ids = {
        int(number) for number in re.findall(r"(?m)^(\d+)\.\s+", skill_bank)
    }
    if not source_skill_ids:
        raise ValueError("No numbered source skills found")
    evidence = build_mined_evidence(args.mined_evidence)
    evidence_source = args.mined_evidence
    if int(evidence.get("sample_count") or 0) != args.expected_evidence_count:
        raise ValueError(
            "Evidence sample count mismatch: "
            f"expected {args.expected_evidence_count}, got {evidence.get('sample_count')}"
        )
    template = args.template.read_text(encoding="utf-8")
    source_prompt = args.s0.read_text(encoding="utf-8")
    source_skill_snapshot = "# Planning Skills\n\n" + skill_bank + "\n"
    evidence_snapshot = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    run_exists = args.output.exists() and any(args.output.iterdir())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "teacher_model": args.model,
        "source_skill_bank": str(args.s0.resolve()),
        "reused_evidence": str(evidence_source.resolve()),
        "generation_prompt": str(args.template.resolve()),
        "evidence_sample_count": int(evidence["sample_count"]),
        "candidate_names": list(STYLES),
        "student_evaluation_run": False,
    }
    manifest_snapshot = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    lock_text_snapshot(
        args.output / "source_skill_bank.txt",
        source_skill_snapshot,
        label="source Skill snapshot",
        require_existing=run_exists,
    )
    lock_text_snapshot(
        args.output / "source_prompt.txt",
        source_prompt,
        label="source Prompt snapshot",
        require_existing=run_exists,
    )
    lock_text_snapshot(
        args.output / "evidence.json",
        evidence_snapshot,
        label="evidence snapshot",
        require_existing=run_exists,
    )
    lock_text_snapshot(
        args.output / "manifest.json",
        manifest_snapshot,
        label="candidate manifest",
        require_existing=run_exists,
    )

    requests: dict[str, str] = {}
    pending: list[str] = []
    for name, style in STYLES.items():
        requests[name] = render(
            template,
            candidate_name=name,
            candidate_style=style,
            skill_bank=skill_bank,
            evidence=json.dumps(evidence, ensure_ascii=False, indent=2),
        )
        candidate_dir = args.output / name
        candidate_exists = candidate_dir.exists() and any(candidate_dir.iterdir())
        candidate_dir.mkdir(parents=True, exist_ok=True)
        lock_text_snapshot(
            candidate_dir / "request.txt",
            requests[name],
            label=f"{name} request",
            require_existing=candidate_exists,
        )
        if candidate_is_complete(
            candidate_dir,
            name=name,
            model=args.model,
            source_skill_ids=source_skill_ids,
            source_prompt=source_prompt,
        ):
            print(f"{name}: cached generated candidate", flush=True)
        else:
            pending.append(name)

    if args.dry_run:
        print(f"Prepared 3 candidate requests in {args.output}")
        return

    if not pending:
        print("All generated candidates are complete; no teacher API call required.", flush=True)
        return

    clients = ThreadClients(args.timeout)
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
        futures = {
            executor.submit(
                call_one,
                clients=clients,
                model=args.model,
                prompt=requests[name],
                name=name,
                retries=args.retries,
                output=args.output,
                source_skill_ids=source_skill_ids,
            ): name
            for name in pending
        }
        failed = False
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
