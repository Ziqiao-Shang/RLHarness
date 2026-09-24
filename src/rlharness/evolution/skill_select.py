"""Select a validated Skill candidate and lock it to round-two training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rlharness.common.api_client import extract_json_obj, chat_text, openlux_client
from rlharness.common.config import ROOT


DOMAINS = ("metromap", "travelmap")
CANDIDATE_NAMES = (
    "candidate_a_conservative",
    "candidate_b_stage_based",
    "candidate_c_compact",
)


def read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected one JSON object in {label}: {path}")
    return value


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_atomic(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def default_evolution_run(domain: str) -> Path:
    if domain == "metromap":
        return ROOT / "artifacts/skill_evolution"
    return ROOT / "artifacts/travelmap/skill_evolution"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _selected_metrics(review: dict[str, Any], candidate: str) -> dict[str, Any]:
    rows = review.get("candidates")
    if not isinstance(rows, list):
        raise SystemExit("candidate_review.json has no candidate list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("candidate") == candidate]
    if len(matches) != 1:
        raise SystemExit(f"Candidate review does not contain exactly one {candidate} entry")
    return matches[0]


def select_candidate(
    *,
    domain: str,
    candidate: str,
    run_dir: Path,
    output_dir: Path,
    replace: bool = False,
    selection_mode: str = "manual_override",
    selection_model: str | None = None,
    selection_reason: str | None = None,
) -> Path:
    """Validate and persist one already selected candidate."""
    if domain not in DOMAINS:
        raise SystemExit(f"Unsupported domain: {domain}")
    if candidate not in CANDIDATE_NAMES:
        raise SystemExit(f"Unsupported candidate: {candidate}")

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    candidate_dir = run_dir / "candidates" / candidate
    validation_dir = run_dir / "validation_qwen35plus" / candidate
    review_path = run_dir / "validation_qwen35plus/candidate_review.json"
    prompt_path = candidate_dir / "full_prompt.txt"
    candidate_path = candidate_dir / "candidate.json"
    skills_path = candidate_dir / "planning_skills.txt"
    rewrite_path = candidate_dir / "fewshot_rewrite.validation.json"
    evaluation_path = validation_dir / "evaluation_manifest.json"
    summary_path = validation_dir / "predictions.summary.json"
    predictions_path = validation_dir / "predictions.jsonl"

    required = {
        "candidate full prompt": prompt_path,
        "candidate JSON": candidate_path,
        "candidate Skill Bank": skills_path,
        "few-shot validation": rewrite_path,
        "candidate review": review_path,
        "evaluation manifest": evaluation_path,
        "evaluation summary": summary_path,
        "evaluation predictions": predictions_path,
    }
    for label, path in required.items():
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")

    review = read_object(review_path, "candidate review")
    metrics = _selected_metrics(review, candidate)
    summary = read_object(summary_path, "evaluation summary")
    expected_metrics = {
        "count": int(summary.get("count") or 0),
        "hard_correct": int(summary.get("hard_correct") or 0),
        "exact_accuracy": int(summary.get("hard_correct") or 0)
        / max(int(summary.get("count") or 0), 1),
        "all_acc": float(summary.get("all_acc") or 0.0),
        "part_acc": float(summary.get("part_acc") or 0.0),
        "format_rate": float(summary.get("format_rate") or 0.0),
        "truncated": int(summary.get("truncated") or 0),
    }
    for name, expected in expected_metrics.items():
        if metrics.get(name) != expected:
            raise SystemExit(
                f"Candidate review metric {name} is stale for {candidate}: "
                f"review={metrics.get(name)!r}, summary={expected!r}"
            )

    rewrite = read_object(rewrite_path, "few-shot validation")
    if rewrite.get("valid") is not True:
        raise SystemExit(f"Few-shot validation did not pass for {candidate}")
    if rewrite.get("required_procedure_preserved") is not True:
        raise SystemExit(f"Required Procedure was not preserved for {candidate}")

    evaluation = read_object(evaluation_path, "evaluation manifest")
    if evaluation.get("candidate") != candidate:
        raise SystemExit("Evaluation manifest candidate does not match the selected candidate")
    if evaluation.get("domain") != domain:
        raise SystemExit("Evaluation manifest domain does not match the selected domain")
    if Path(str(evaluation.get("prompt") or "")).resolve() != prompt_path.resolve():
        raise SystemExit("Selected prompt differs from the prompt evaluated on Val100")
    prompt_snapshot = Path(str(evaluation.get("prompt_snapshot") or ""))
    if not prompt_snapshot.is_file():
        raise SystemExit("Evaluation manifest has no readable Val100 Prompt snapshot")
    if prompt_snapshot.read_bytes() != prompt_path.read_bytes():
        raise SystemExit("Selected prompt content differs from the Prompt evaluated on Val100")

    sources = {
        "prompt": {"path": _relative(prompt_path, run_dir)},
        "candidate": {"path": _relative(candidate_path, run_dir)},
        "planning_skills": {"path": _relative(skills_path, run_dir)},
        "fewshot_validation": {"path": _relative(rewrite_path, run_dir)},
        "candidate_review": {"path": _relative(review_path, run_dir)},
        "evaluation_manifest": {"path": _relative(evaluation_path, run_dir)},
        "evaluation_summary": {"path": _relative(summary_path, run_dir)},
        "evaluation_predictions": {"path": _relative(predictions_path, run_dir)},
        "evaluated_prompt": {"path": _relative(prompt_snapshot, run_dir)},
    }
    selection = {
        "schema_version": 1,
        "selection_mode": selection_mode,
        "domain": domain,
        "candidate": candidate,
        "metrics": metrics,
        "sources": sources,
    }
    if selection_model is not None:
        selection["selection_model"] = selection_model
    if selection_reason is not None:
        selection["reason"] = selection_reason
    selection_path = output_dir / "selection.json"
    if selection_path.exists():
        existing = read_object(selection_path, "existing selection")
        if existing != selection and not replace:
            raise SystemExit(
                f"A different candidate selection already exists at {selection_path}. "
                "Use --replace only when deliberately replacing that decision."
            )
    elif output_dir.exists() and any(output_dir.iterdir()) and not replace:
        raise SystemExit(
            f"Selection output directory is not empty and has no selection record: {output_dir}"
        )

    write_atomic(output_dir / "full_prompt.txt", prompt_path.read_bytes())
    write_atomic(output_dir / "candidate.json", candidate_path.read_bytes())
    write_atomic(output_dir / "planning_skills.txt", skills_path.read_bytes())
    write_json_atomic(selection_path, selection)
    print(f"Selected candidate: {candidate}")
    print(f"Locked prompt: {output_dir / 'full_prompt.txt'}")
    print(f"Selection record: {selection_path}")
    return selection_path


def auto_select_candidate(
    *,
    domain: str,
    run_dir: Path,
    output_dir: Path,
    model: str = "gpt-5.6-sol",
    template_path: Path | None = None,
    timeout: float = 300.0,
    max_tokens: int = 2048,
    replace: bool = False,
    client: Any | None = None,
) -> Path:
    """Ask GPT-5.6-Sol to choose from the existing fixed-Val100 report."""
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    review_path = run_dir / "validation_qwen35plus/candidate_review.json"
    review = read_object(review_path, "candidate review")
    rows = review.get("candidates")
    names = {
        row.get("candidate")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("candidate"), str)
    } if isinstance(rows, list) else set()
    if names != set(CANDIDATE_NAMES):
        raise SystemExit(
            f"Candidate review must contain exactly {list(CANDIDATE_NAMES)}, found {sorted(names)}"
        )

    selection_path = output_dir / "selection.json"
    if selection_path.is_file() and not replace:
        existing = read_object(selection_path, "existing selection")
        candidate = existing.get("candidate")
        selected_prompt = output_dir / "full_prompt.txt"
        candidate_prompt = run_dir / "candidates" / str(candidate) / "full_prompt.txt"
        if (
            existing.get("selection_mode") == "gpt56sol_automatic"
            and existing.get("selection_model") == model
            and existing.get("domain") == domain
            and candidate in CANDIDATE_NAMES
            and selected_prompt.is_file()
            and candidate_prompt.is_file()
            and selected_prompt.read_bytes() == candidate_prompt.read_bytes()
        ):
            print(f"Reusing existing automatic selection: {selection_path}")
            return selection_path
        raise SystemExit(
            f"Existing selection does not match this automatic selection run: {selection_path}. "
            "Use --replace to run the selector again."
        )

    template_path = (
        template_path or ROOT / "prompts" / "evolution" / domain / "select.txt"
    ).resolve()
    if not template_path.is_file():
        raise SystemExit(f"Missing candidate selection Prompt: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    marker = "{candidate_review}"
    if template.count(marker) != 1:
        raise SystemExit(f"Selection Prompt must contain exactly one {marker}: {template_path}")
    rendered = template.replace(
        marker,
        json.dumps(review, ensure_ascii=False, indent=2),
    )

    selector_client = client or openlux_client(timeout=timeout)
    response = chat_text(
        selector_client,
        model,
        [{"role": "user", "content": rendered}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    decision = extract_json_obj(response)
    if not isinstance(decision, dict):
        raise SystemExit("GPT-5.6-Sol selector did not return one JSON object")
    candidate = decision.get("selected_candidate")
    reason = decision.get("reason")
    if candidate not in CANDIDATE_NAMES:
        raise SystemExit(f"GPT-5.6-Sol selected an unknown candidate: {candidate!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise SystemExit("GPT-5.6-Sol selector returned an empty reason")

    return select_candidate(
        domain=domain,
        candidate=candidate,
        run_dir=run_dir,
        output_dir=output_dir,
        replace=replace,
        selection_mode="gpt56sol_automatic",
        selection_model=model,
        selection_reason=reason.strip(),
    )


def lock_training_prompt(
    *,
    stage: str,
    domain: str,
    prompt: Path,
    output_dir: Path,
    selection_manifest: Path | None = None,
) -> Path:
    """Freeze the exact prompt consumed by one RL training directory."""
    if stage not in {"round1", "round2"}:
        raise SystemExit(f"Unsupported training stage: {stage}")
    if domain not in DOMAINS:
        raise SystemExit(f"Unsupported domain: {domain}")
    prompt = prompt.resolve()
    output_dir = output_dir.resolve()
    if not prompt.is_file():
        raise SystemExit(f"Missing {stage} prompt: {prompt}")
    prompt_bytes = prompt.read_bytes()

    selection_record: dict[str, Any] | None = None
    if selection_manifest is not None:
        selection_manifest = selection_manifest.resolve()
        selection_record = read_object(selection_manifest, "selection manifest")
        if selection_record.get("domain") != domain:
            raise SystemExit(f"Selection manifest domain does not match {stage} training")
        selected_prompt = selection_manifest.parent / "full_prompt.txt"
        if not selected_prompt.is_file() or selected_prompt.read_bytes() != prompt_bytes:
            raise SystemExit(f"{stage} prompt does not match the selected prompt")

    manifest = {
        "schema_version": 1,
        "stage": stage,
        "domain": domain,
        "source_prompt": str(prompt),
        "selection": (
            {
                "path": str(selection_manifest),
                "candidate": selection_record.get("candidate"),
            }
            if selection_manifest is not None and selection_record is not None
            else None
        ),
    }
    manifest_path = output_dir / "prompt_manifest.json"
    snapshot_path = output_dir / "selected_prompt.txt"
    if manifest_path.exists():
        existing = read_object(manifest_path, f"{stage} prompt manifest")
        if existing != manifest:
            raise SystemExit(
                f"{stage} output is already locked to another prompt: {manifest_path}. "
                "Use a new RL_OUTPUT_DIR."
            )
    elif output_dir.exists() and any((output_dir / name).exists() for name in ("checkpoints", "rollouts", "validation")):
        raise SystemExit(
            f"Existing {stage} outputs have no prompt manifest: {output_dir}. "
            "Do not mix an old run with a newly selected prompt."
        )
    if snapshot_path.exists() and snapshot_path.read_bytes() != prompt_bytes:
        raise SystemExit(f"{stage} prompt snapshot conflicts with the requested prompt: {snapshot_path}")

    write_atomic(snapshot_path, prompt_bytes)
    write_json_atomic(manifest_path, manifest)
    print(f"{stage} prompt snapshot: {snapshot_path}")
    return snapshot_path


def lock_round1_prompt(*, domain: str, prompt: Path, output_dir: Path) -> Path:
    return lock_training_prompt(
        stage="round1",
        domain=domain,
        prompt=prompt,
        output_dir=output_dir,
    )


def lock_round2_prompt(
    *,
    domain: str,
    prompt: Path,
    output_dir: Path,
    selection_manifest: Path | None = None,
) -> Path:
    return lock_training_prompt(
        stage="round2",
        domain=domain,
        prompt=prompt,
        output_dir=output_dir,
        selection_manifest=selection_manifest,
    )


def verify_round2_prompt(*, domain: str, output_dir: Path) -> Path:
    """Verify and return the prompt snapshot paired with a round-two run."""
    output_dir = output_dir.resolve()
    manifest = read_object(output_dir / "prompt_manifest.json", "round-two prompt manifest")
    snapshot = output_dir / "selected_prompt.txt"
    if not snapshot.is_file():
        raise SystemExit(f"Missing round-two prompt snapshot: {snapshot}")
    if manifest.get("domain") != domain:
        raise SystemExit("Round-two prompt manifest domain does not match evaluation")
    if manifest.get("stage") != "round2":
        raise SystemExit("Prompt manifest does not belong to round-two training")
    return snapshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="Select and record one validated candidate")
    select.add_argument("--domain", choices=DOMAINS, default="metromap")
    select.add_argument(
        "--candidate",
        choices=CANDIDATE_NAMES,
        help="Manual override; omit to use the formal GPT-5.6-Sol selector.",
    )
    select.add_argument("--run-dir", type=Path)
    select.add_argument("--output-dir", type=Path)
    select.add_argument("--model", default="gpt-5.6-sol")
    select.add_argument("--template", type=Path)
    select.add_argument("--timeout", type=float, default=300.0)
    select.add_argument("--max-tokens", type=int, default=2048)
    select.add_argument(
        "--confirm-reviewed",
        action="store_true",
        help="Confirm that accuracy and per-Skill usage were both reviewed by a human.",
    )
    select.add_argument("--replace", action="store_true")

    lock = subparsers.add_parser("lock-round2", help="Freeze a round-two training prompt")
    lock.add_argument("--domain", choices=DOMAINS, required=True)
    lock.add_argument("--prompt", type=Path, required=True)
    lock.add_argument("--output-dir", type=Path, required=True)
    lock.add_argument("--selection-manifest", type=Path)

    lock_round1 = subparsers.add_parser("lock-round1", help="Freeze a round-one training prompt")
    lock_round1.add_argument("--domain", choices=DOMAINS, required=True)
    lock_round1.add_argument("--prompt", type=Path, required=True)
    lock_round1.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify-round2", help="Verify a round-two prompt snapshot")
    verify.add_argument("--domain", choices=DOMAINS, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "select":
        run_dir = (args.run_dir or default_evolution_run(args.domain)).resolve()
        output_dir = (args.output_dir or run_dir / "selected").resolve()
        if args.candidate:
            if not args.confirm_reviewed:
                raise SystemExit(
                    "Manual override requires --confirm-reviewed. Omit --candidate to use "
                    "the formal GPT-5.6-Sol selector."
                )
            select_candidate(
                domain=args.domain,
                candidate=args.candidate,
                run_dir=run_dir,
                output_dir=output_dir,
                replace=args.replace,
                selection_mode="manual_override",
                selection_reason="Explicit human override after reviewing Val100 metrics.",
            )
        else:
            auto_select_candidate(
                domain=args.domain,
                run_dir=run_dir,
                output_dir=output_dir,
                model=args.model,
                template_path=args.template,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                replace=args.replace,
            )
    elif args.command == "lock-round1":
        lock_round1_prompt(
            domain=args.domain,
            prompt=args.prompt,
            output_dir=args.output_dir,
        )
    elif args.command == "lock-round2":
        lock_round2_prompt(
            domain=args.domain,
            prompt=args.prompt,
            output_dir=args.output_dir,
            selection_manifest=args.selection_manifest,
        )
    else:
        snapshot = verify_round2_prompt(domain=args.domain, output_dir=args.output_dir)
        print(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
