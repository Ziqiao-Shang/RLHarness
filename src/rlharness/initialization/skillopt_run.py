"""Create a seed prompt, then optimize it with SkillOpt on Train480/Val100."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path

from rlharness.common.config import ROOT
from rlharness.initialization.state import read_json, write_json, write_text
from rlharness.initialization import seed as seed_pipeline

from skillopt.config import flatten_config, load_config
from skillopt.engine.trainer import ReflACTTrainer
from skillopt.model import configure_azure_openai
from rlharness.initialization.skillopt_adapter import LockedMapLoader, MapSkillOptAdapter, skill_strings


def lock_run(output: Path, record: dict) -> None:
    path = output / "input_manifest.json"
    if path.exists():
        if read_json(path) != record:
            raise ValueError("Inputs/models/code changed; use a new output directory")
    elif output.exists() and any(p.name != ".pipeline.lock" for p in output.iterdir()):
        raise ValueError("Refusing to reuse an output directory without a matching input manifest")
    write_json(path, record)


def input_record(cfg: dict, loader: LockedMapLoader, prompt: str) -> dict:
    assets: list[str] = []
    from rlharness.common.prompt_build import resolve_image, resolve_vertex_table_path
    for row in loader.rows.values():
        for name in (resolve_image(row.get("figure")), resolve_vertex_table_path(row)):
            if not name or not Path(name).is_file():
                raise ValueError(f"Missing dataset asset for {row['id']}")
            resolved = str(Path(name).resolve())
            if resolved not in assets:
                assets.append(resolved)
    return {"config": cfg, "prompt": prompt, "assets": sorted(assets),
            "batches": loader.batches,
            "validation_ids": [r["id"] for r in loader.val_items],
            "split_manifest": loader.manifest}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("metromap", "travelmap"), default="metromap")
    parser.add_argument("--split-dir", type=Path, required=True, help="Prepared Train480 split directory")
    parser.add_argument("--initial-prompt", type=Path, help="Skip seed creation and optimize this complete prompt")
    parser.add_argument("--demo-ids", nargs=2, help="Seed examples; default: first two locked Train480 IDs")
    parser.add_argument("--seed-max-tokens", type=int, default=8192)
    parser.add_argument("--seed-repair-attempts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-model", default="gemini-3.5-flash")
    parser.add_argument("--teacher-model", default="gpt-5.6-sol")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=12,
        help="Stop after this many SkillOpt steps; the locked full run uses 12",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analyst-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Validate data/assets/config; never call APIs")
    args = parser.parse_args(argv)
    if min(args.workers, args.analyst_workers, args.seed_max_tokens,
           args.seed_repair_attempts, args.max_train_steps) < 1:
        parser.error("Worker, token, and retry counts must be positive")
    if args.max_train_steps > 12:
        parser.error("The locked Train480 protocol has at most 12 training steps")
    if args.initial_prompt and args.demo_ids:
        parser.error("--demo-ids only applies when creating a new seed")
    args.output_dir = args.output_dir.resolve()
    args.split_dir = args.split_dir.resolve()
    return args


def run(args, *, seed_backend=None, trainer_cls=None):
    cfg = flatten_config(load_config(str(ROOT / "configs" / "skillopt.yaml")))
    optimization_dir = args.output_dir / "optimization"
    initial = (args.initial_prompt.resolve() if args.initial_prompt else
               args.output_dir / "seed" / "versions" / "v0000" / "full_prompt.txt")
    cfg.update(env=args.domain, skill_init=str(initial), out_root=str(optimization_dir),
               target_model=args.student_model, optimizer_model=args.teacher_model,
               rollout_workers=args.workers, analyst_workers=args.analyst_workers,
               max_train_steps=args.max_train_steps,
               azure_openai_endpoint=os.environ.get("OPENLUX_BASE_URL", "https://api.openlux.ai/v1"))
    loader = LockedMapLoader(args.domain, args.split_dir.resolve())
    cfg["seed"] = int(loader.manifest["seed"])
    seed_args = None
    if args.initial_prompt:
        source_text = initial.read_text(encoding="utf-8")
        adapter = MapSkillOptAdapter(args.domain, loader)
        adapter.setup(cfg)
        valid, reasons = adapter.validate_candidate_skill(source_text)
        if not valid:
            raise ValueError(f"Invalid initial prompt: {reasons}")
        seed_manifest = {"mode": "provided_prompt"}
    else:
        demos = args.demo_ids or [r["id"] for r in loader.train_items[:2]]
        seed_args = seed_pipeline.parse_args([
            "--domain", args.domain,
            "--train-ids", str(args.split_dir / "train_sample_ids.json"),
            "--validation-ids", str(args.split_dir / "validation_sample_ids.json"),
            "--test-ids", str(args.split_dir / "test_sample_ids.json"),
            "--demo-ids", *demos, "--output-dir", str(args.output_dir / "seed"),
            "--teacher-model", args.teacher_model,
            "--max-tokens", str(args.seed_max_tokens),
            "--repair-attempts", str(args.seed_repair_attempts),
        ])
        _, demo_rows, source_text, _, seed_manifest = seed_pipeline.prepare(seed_args)
        print(f"[initialize] create 15 skills + usage + two verified examples: {demos}")
    record = input_record(cfg, loader, source_text)
    record["seed_initialization"] = seed_manifest
    record["backend"] = "api" if seed_backend is None and trainer_cls is None else "simulation"
    # Explicit demonstrations may be outside Train480; include their assets in the run lock.
    if seed_args is not None:
        from rlharness.initialization.execution import bundle_for
        record["demo_ids"] = [row["sample_id"] for row in demo_rows]
        for row in demo_rows:
            bundle = bundle_for(row, "{question}")
            for name in (bundle["image_path"], bundle["table_path"]):
                path = str(Path(name).resolve())
                if path not in record["assets"]:
                    record["assets"].append(path)
        record["assets"].sort()
    print(f"[SkillOpt] {args.domain}: Train480, 12 x 40, fixed Val100, no test evaluation")
    print(f"[models] student={args.student_model}; teacher={args.teacher_model}")
    print(f"[prompt] {initial}")
    print(f"[map overlap] {loader.manifest.get('figure_overlap', {})}")
    if args.dry_run:
        print("[dry-run OK] inputs/assets validated; no API calls and no training outputs written")
        return record
    key = os.environ.get("OPENLUX_API_KEY")
    if record["backend"] == "api" and not key:
        raise ValueError("Set OPENLUX_API_KEY in the environment (never in code/config)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_run(args.output_dir, record)
        if seed_args is not None:
            state = seed_pipeline.run(seed_args, backend=seed_backend)
            if state["version"] != 0 or state["status"] != "seeded":
                raise ValueError("Expected one unoptimized seed before SkillOpt")
        adapter = MapSkillOptAdapter(args.domain, loader)
        adapter.setup(cfg)
        valid, reasons = adapter.validate_candidate_skill(adapter.baseline)
        if not valid:
            raise ValueError(f"Generated seed violates SkillOpt contract: {reasons}")
        lock_run(optimization_dir, input_record(cfg, loader, adapter.baseline))
        # Credentials never enter cfg, manifests, CLI arguments, or saved summaries.
        configure_azure_openai(api_key=key, endpoint=cfg["azure_openai_endpoint"],
                               auth_mode="openai_compatible")
        summary = (trainer_cls or ReflACTTrainer)(cfg, adapter).train()
        best = (optimization_dir / "best_skill.md").read_text(encoding="utf-8")
        write_text(args.output_dir / "best" / "full_prompt.txt", best)
        write_json(args.output_dir / "best" / "skills.json", {"skills": skill_strings(best)})
        write_json(args.output_dir / "best" / "manifest.json", {
            "domain": args.domain, "selection_split": "validation", "best_step": summary["best_step"],
            "test_used": False, "backend": record["backend"],
        })
    print(f"[best-on-val] {args.output_dir / 'best' / 'full_prompt.txt'}")
    return summary


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
