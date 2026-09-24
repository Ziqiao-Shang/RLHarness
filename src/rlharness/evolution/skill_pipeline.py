"""Run the complete post-RL skill-evolution and candidate-validation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TextIO

from rlharness.common.config import ROOT


CANDIDATE_NAMES = (
    "candidate_a_conservative",
    "candidate_b_stage_based",
    "candidate_c_compact",
)

MODULES = {
    "eval_local": "rlharness.eval.eval_local",
    "eval_api": "rlharness.eval.eval_api",
    "eval_merge": "rlharness.eval.eval_merge",
    "skill_evidence": "rlharness.evolution.skill_evidence",
    "skill_generate": "rlharness.evolution.skill_generate",
    "skill_prune": "rlharness.evolution.skill_prune",
    "skill_report": "rlharness.evolution.skill_report",
    "skill_rewrite": "rlharness.evolution.skill_rewrite",
    "skill_select": "rlharness.evolution.skill_select",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("metromap", "travelmap"), default="metromap")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="Current full prompt; defaults to prompts/student/<domain>/original.txt.",
    )
    parser.add_argument(
        "--sample-ids",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--validation-ids",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--gpu-ids", nargs="+", default=os.environ.get("GPU_IDS", "0 1 2 3").split())
    parser.add_argument("--teacher-model", default="gpt-5.6-sol")
    parser.add_argument("--validation-model", default="qwen3.5-plus")
    parser.add_argument("--selection-model", default="gpt-5.6-sol")
    parser.add_argument("--validation-split", choices=("train", "test"), default="train")
    parser.add_argument("--analysis-template", type=Path, default=None)
    parser.add_argument("--generation-template", type=Path, default=None)
    parser.add_argument("--rewrite-template", type=Path, default=None)
    parser.add_argument("--selection-template", type=Path, default=None)
    parser.add_argument("--min-usage-rate", type=float, default=0.005)
    parser.add_argument("--force-keep-skills", default="")
    parser.add_argument("--skip-rollout", action="store_true")
    return parser.parse_args()


def module_command(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", MODULES[module], *(str(value) for value in args)]


def run(command: list[str], *, stdout: TextIO | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, stdout=stdout, stderr=subprocess.STDOUT if stdout else None)


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def require_locked_evolution_ids(domain: str, train_path: Path, validation_path: Path) -> None:
    """Keep post-RL evidence and candidate selection inside Train1600/Val100."""
    reference = ROOT / "data" / "reference" / domain / "splits"
    supplied_train = json.loads(train_path.read_text(encoding="utf-8"))
    supplied_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    locked_train = json.loads((reference / "train_sample_ids.json").read_text(encoding="utf-8"))
    locked_validation = json.loads(
        (reference / "validation_sample_ids.json").read_text(encoding="utf-8")
    )
    if supplied_train != locked_train:
        raise SystemExit("Skill evolution must use the exact locked Train1600 IDs")
    if supplied_validation != locked_validation:
        raise SystemExit("Candidate selection must use the exact locked Val100 IDs")


def check_or_create_manifest(
    path: Path,
    payload: dict[str, object],
    *,
    label: str,
    require_existing: bool = False,
) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SystemExit(
                f"{label} configuration differs from the existing manifest: {path}. "
                "Use a new output directory instead of mixing runs."
            )
        return
    if require_existing:
        raise SystemExit(f"Missing {label} manifest required for safe reuse: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_api_key() -> None:
    if not os.environ.get("OPENLUX_API_KEY"):
        raise SystemExit("Set OPENLUX_API_KEY before evolving skills")


def generate_rollouts(args: argparse.Namespace) -> Path:
    output = args.source_dir
    shards_dir = output / "shards"
    logs_dir = output / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = json.loads(args.sample_ids.read_text(encoding="utf-8"))
    total = len(sample_ids)
    if not total:
        raise SystemExit(f"No rollout sample IDs in {args.sample_ids}")
    manifest_path = output / "rollout_manifest.json"
    rollout_manifest = {
        "schema_version": 1,
        "domain": args.domain,
        "split": "train",
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "prompt": str(args.prompt.resolve()),
        "sample_ids": str(args.sample_ids.resolve()),
    }
    if not manifest_path.is_file() and (
        (output / "train_predictions.jsonl").exists()
        or any(shards_dir.glob("*.jsonl"))
    ):
        raise SystemExit(
            f"Existing rollout files have no provenance manifest: {output}. "
            "Use a new --source-dir."
        )
    check_or_create_manifest(manifest_path, rollout_manifest, label="rollout")

    processes: list[tuple[subprocess.Popen[bytes], TextIO]] = []
    shards: list[Path] = []
    try:
        for index, gpu in enumerate(args.gpu_ids):
            start = index * total // len(args.gpu_ids)
            end = (index + 1) * total // len(args.gpu_ids)
            shard = shards_dir / f"gpu_{index}.jsonl"
            log_handle = (logs_dir / f"gpu_{index}.log").open("w", encoding="utf-8")
            command = module_command(
                "eval_local",
                "--domain", args.domain,
                "--split", "train",
                "--sample-ids-from", args.sample_ids,
                "--prompt-template", args.prompt,
                "--model", args.model,
                "--adapter", args.adapter,
                "--output", shard,
                "--start", start,
                "--limit", end - start,
                "--no-thinking",
                "--visible-reasoning",
                "--max-new-tokens", 4096,
                "--max-model-len", 32768,
                "--image-max-pixels", 1_000_000,
                "--table-max-chars", 20_000,
                "--batch-size", 16,
                "--gpu-memory-utilization", 0.70,
                "--skip-mm-profiling",
                "--resume",
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            print("+", " ".join(command), f"[GPU {gpu}]", flush=True)
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            processes.append((process, log_handle))
            shards.append(shard)

        failures = []
        for process, log_handle in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code:
                failures.append(return_code)
        if failures:
            raise SystemExit(f"Rollout generation failed; inspect {logs_dir}")
    except BaseException:
        for process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
            if not log_handle.closed:
                log_handle.close()
        raise

    predictions = output / "train_predictions.jsonl"
    run(
        module_command(
            "eval_merge",
            "--domain", args.domain,
            "--split", "train",
            "--sample-ids-from", args.sample_ids,
            "--output", predictions,
            *shards,
        )
    )
    return predictions


def validate_rewrite(candidate_dir: Path) -> None:
    path = candidate_dir / "fewshot_rewrite.validation.json"
    require(path, "few-shot validation")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("valid") is not True:
        raise SystemExit(f"Few-shot rewrite validation did not pass: {path}")
    if record.get("required_procedure_preserved") is not True:
        raise SystemExit(f"Required Procedure was not preserved: {path}")


def lock_evaluated_prompt(prompt: Path, output: Path) -> Path:
    """Bind resumable Val predictions to the exact Prompt bytes they evaluated."""
    snapshot = output.parent / "evaluated_prompt.txt"
    prompt_bytes = prompt.read_bytes()
    if output.exists() and not snapshot.is_file():
        raise SystemExit(
            f"Existing candidate predictions have no Prompt snapshot: {output}. "
            "Use a new --run-dir."
        )
    if snapshot.is_file() and snapshot.read_bytes() != prompt_bytes:
        raise SystemExit(
            f"Candidate Prompt differs from the existing Val predictions: {snapshot}. "
            "Use a new --run-dir."
        )
    if not snapshot.is_file():
        temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
        temporary.write_bytes(prompt_bytes)
        temporary.replace(snapshot)
    return snapshot


def validate_candidates(args: argparse.Namespace, candidates: Path, validation: Path) -> None:
    for name in CANDIDATE_NAMES:
        candidate_dir = candidates / name
        prompt = candidate_dir / "full_prompt.txt"
        require(prompt, "candidate prompt")
        validate_rewrite(candidate_dir)
        output = validation / name / "predictions.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        prompt_snapshot = lock_evaluated_prompt(prompt, output)
        evaluation_manifest = {
            "schema_version": 1,
            "candidate": name,
            "domain": args.domain,
            "split": args.validation_split,
            "model": args.validation_model,
            "prompt": str(prompt.resolve()),
            "prompt_snapshot": str(prompt_snapshot.resolve()),
            "sample_ids": str(args.validation_ids.resolve()),
        }
        evaluation_manifest_path = output.parent / "evaluation_manifest.json"
        if output.exists() and not evaluation_manifest_path.is_file():
            raise SystemExit(
                f"Existing candidate predictions have no provenance manifest: {output}. "
                "Use a new --run-dir."
            )
        check_or_create_manifest(
            evaluation_manifest_path,
            evaluation_manifest,
            label=f"{name} validation",
        )
        run(
            module_command(
                "eval_api",
                "--domain", args.domain,
                "--split", args.validation_split,
                "--model", args.validation_model,
                "--no-enable-thinking",
                "--sample-ids-from", args.validation_ids,
                "--prompt-template", prompt,
                "--output", output,
                "--workers", 100,
                "--temperature", 0,
                "--max-tokens", 8192,
                "--timeout", 180,
                "--retries", 3,
                "--table-max-chars", 20_000,
                "--image-max-pixels", 1_000_000,
                "--seed", 0,
                "--resume",
            )
        )

    run(
        module_command(
            "skill_report",
            "--candidate-root", candidates,
            "--evaluation-root", validation,
            "--output", validation / "candidate_review.json",
        )
    )


def main() -> None:
    args = parse_args()
    if args.domain == "metromap":
        artifact_root = ROOT / "artifacts"
        split_root = ROOT / "data/generated/splits"
    else:
        artifact_root = ROOT / "artifacts/travelmap"
        split_root = ROOT / "data/generated/travelmap/splits"
    args.model = args.model or artifact_root / "sft_merged"
    args.adapter = args.adapter or artifact_root / "grpo_round1/checkpoints/global_step_400/adapter"
    args.sample_ids = args.sample_ids or split_root / "train_sample_ids.json"
    args.validation_ids = args.validation_ids or split_root / "validation_sample_ids.json"
    args.source_dir = args.source_dir or artifact_root / "skill_evolution_source"
    args.run_dir = args.run_dir or artifact_root / "skill_evolution"
    ensure_api_key()
    args.prompt = args.prompt or ROOT / "prompts/student" / args.domain / "original.txt"
    prompt_root = ROOT / "prompts/evolution" / args.domain
    analysis_template = args.analysis_template or prompt_root / "analyze.txt"
    generation_template = args.generation_template or prompt_root / "generate.txt"
    rewrite_template = args.rewrite_template or prompt_root / "rewrite.txt"
    selection_template = args.selection_template or prompt_root / "select.txt"
    for path, label in (
        (args.model / "config.json", "SFT model"),
        (args.adapter / "adapter_config.json", "round-1 adapter"),
        (args.prompt, "original prompt"),
        (args.sample_ids, "training sample IDs"),
        (args.validation_ids, "validation sample IDs"),
        (analysis_template, f"{args.domain} analysis prompt"),
        (generation_template, f"{args.domain} generation prompt"),
        (rewrite_template, f"{args.domain} rewrite prompt"),
        (selection_template, f"{args.domain} selection prompt"),
    ):
        require(path, label)
    if args.validation_split != "train":
        raise SystemExit("Candidate selection is restricted to the locked Val100 from the train source")
    require_locked_evolution_ids(args.domain, args.sample_ids, args.validation_ids)

    predictions = args.source_dir / "train_predictions.jsonl"
    if not args.skip_rollout:
        predictions = generate_rollouts(args)
    else:
        check_or_create_manifest(
            args.source_dir / "rollout_manifest.json",
            {
                "schema_version": 1,
                "domain": args.domain,
                "split": "train",
                "model": str(args.model.resolve()),
                "adapter": str(args.adapter.resolve()),
                "prompt": str(args.prompt.resolve()),
                "sample_ids": str(args.sample_ids.resolve()),
            },
            label="rollout",
            require_existing=True,
        )
    require(predictions, "skill-evolution rollouts")

    pruning = args.run_dir / "pruning"
    pruned_prompt = pruning / "pruned_prompt.txt"
    usage_report = pruning / "usage_report.json"
    run(
        module_command(
            "skill_prune",
            "--prompt", args.prompt,
            "--rollouts", predictions,
            "--output-prompt", pruned_prompt,
            "--output-report", usage_report,
            "--min-usage-rate", args.min_usage_rate,
            "--keep", args.force_keep_skills,
        )
    )

    evidence = args.run_dir / "evidence128"
    shared_evidence_args = (
        "--s0", pruned_prompt,
        "--rollouts", predictions,
        "--output", evidence,
        "--template", analysis_template,
        "--sample-count", 128,
        "--batch-size", 16,
        "--seed", 20260917,
    )
    run(module_command("skill_evidence", "prepare", *shared_evidence_args))
    run(
        module_command(
            "skill_evidence",
            "mine",
            *shared_evidence_args,
            "--model", args.teacher_model,
            "--workers", 8,
            "--target-reports", 8,
        )
    )

    candidates = args.run_dir / "candidates"
    run(
        module_command(
            "skill_generate",
            "--s0", pruned_prompt,
            "--mined-evidence", evidence,
            "--expected-evidence-count", 128,
            "--model", args.teacher_model,
            "--template", generation_template,
            "--workers", 3,
            "--timeout", 600,
            "--output", candidates,
        )
    )
    run(
        module_command(
            "skill_rewrite",
            "--candidate-root", candidates,
            "--model", args.teacher_model,
            "--template", rewrite_template,
            "--workers", 3,
            "--timeout", 600,
            "--retries", 3,
        )
    )

    validation = args.run_dir / "validation_qwen35plus"
    validate_candidates(args, candidates, validation)
    print(f"Candidate review: {validation / 'candidate_review.json'}")
    run(
        module_command(
            "skill_select",
            "select",
            "--domain", args.domain,
            "--run-dir", args.run_dir,
            "--output-dir", args.run_dir / "selected",
            "--model", args.selection_model,
            "--template", selection_template,
        )
    )
    print(f"Selected Prompt: {args.run_dir / 'selected/full_prompt.txt'}")


if __name__ == "__main__":
    main()
