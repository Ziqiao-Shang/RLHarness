"""Create the initial Skill Bank, usage protocol, and two verified examples."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Any

from rlharness.common.config import ROOT
from rlharness.common.data_load import load_planning
from .execution import APIBackend, bundle_for, example_text, solve
from .state import assemble, read_json, validate_bank, write_json, write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("metromap", "travelmap"), required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--demo-ids", nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default="gpt-5.6-sol")
    parser.add_argument("--repair-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--image-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without API calls")
    return parser.parse_args(argv)


def read_ids(path: Path) -> list[str]:
    values = read_json(path)
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"Expected a nonempty unique sample-ID list: {path}")
    return values


def prepare(args: argparse.Namespace) -> tuple[list[dict], list[dict], str, dict, dict]:
    if args.repair_attempts < 1 or args.max_tokens < 1 or args.timeout <= 0:
        raise ValueError("Invalid retry, token, or timeout setting")
    if args.image_max_pixels < 1:
        raise ValueError("image-max-pixels must be positive")

    train_ids = read_ids(args.train_ids)
    validation_ids = read_ids(args.validation_ids)
    test_ids = read_ids(args.test_ids)
    reference = ROOT / "data" / "reference" / args.domain / "splits"
    locked_train = read_ids(reference / "train_sample_ids.json")
    locked_validation = read_ids(reference / "validation_sample_ids.json")
    locked_test = read_ids(reference / "test_sample_ids.json")
    if len(train_ids) != 480 or not set(train_ids) <= set(locked_train):
        raise ValueError("Seed creation requires Train480 drawn only from locked Train1600")
    if validation_ids != locked_validation or test_ids != locked_test:
        raise ValueError("Seed creation requires the exact locked Val100 and Test400 IDs")
    if len(set(args.demo_ids)) != 2 or not set(args.demo_ids) <= set(locked_train):
        raise ValueError("Two distinct demonstration IDs from locked Train1600 are required")
    if set(args.demo_ids) & (set(validation_ids) | set(test_ids)):
        raise ValueError("Demonstration sample IDs overlap validation/test")

    source_train = {row["sample_id"]: row for row in load_planning(args.domain, "train")}
    source_test = {row["sample_id"]: row for row in load_planning(args.domain, "test")}
    if not set(train_ids + args.demo_ids + validation_ids) <= source_train.keys():
        raise ValueError("Unknown or wrong-source training/validation sample ID")
    if not set(test_ids) <= source_test.keys():
        raise ValueError("Unknown or wrong-source test sample ID")
    rows = [source_train[sample_id] for sample_id in train_ids]
    demos = [source_train[sample_id] for sample_id in args.demo_ids]

    prompt_root = ROOT / "prompts" / "initialization" / args.domain
    frame = (ROOT / "prompts" / "student" / args.domain / "frame.txt").read_text(
        encoding="utf-8"
    )
    templates = {
        name: (prompt_root / f"{name}.txt").read_text(encoding="utf-8")
        for name in ("create_skills", "update_usage", "solve_task")
    }
    for row in demos:
        bundle_for(row, "{question}")

    training_figures = {source_train[sample_id]["figure"] for sample_id in train_ids + args.demo_ids}
    heldout_figures = {
        source_train[sample_id]["figure"] for sample_id in validation_ids
    } | {source_test[sample_id]["figure"] for sample_id in test_ids}
    manifest = {
        "schema": 1,
        "domain": args.domain,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "demo_ids": args.demo_ids,
        "protocol": "seed_then_skillopt",
        "training_demo_heldout_figure_overlap": len(training_figures & heldout_figures),
        "frame": str((ROOT / "prompts" / "student" / args.domain / "frame.txt").resolve()),
        "templates": {
            name: str((prompt_root / f"{name}.txt").resolve()) for name in templates
        },
        "teacher_model": args.teacher_model,
        "repair_attempts": args.repair_attempts,
        "max_tokens": args.max_tokens,
        "image_max_pixels": args.image_max_pixels,
    }
    return rows, demos, frame, templates, manifest


def synchronize(
    backend: Any,
    args: argparse.Namespace,
    bank: dict,
    demos: list[dict],
    frame: str,
    templates: dict,
    folder: Path,
) -> tuple[str, list[dict], str]:
    feedback: list[str] = []
    for attempt in range(args.repair_attempts):
        value = backend.request(
            folder / f"usage_{attempt}.json",
            model=args.teacher_model,
            instruction=templates["update_usage"],
            payload={"skill_bank": bank, "fixed_frame": frame, "feedback": feedback},
            images=[],
        )
        try:
            usage = value["how_to_use"]
            provisional = assemble(frame, bank, usage, [])
            examples = []
            for index, row in enumerate(demos, 1):
                result = solve(
                    backend,
                    row=row,
                    prompt=provisional,
                    skill_count=len(bank["skills"]),
                    model=args.teacher_model,
                    template=templates["solve_task"],
                    folder=folder / f"usage_{attempt}_demo_{index}",
                    attempts=args.repair_attempts,
                )
                if not result["accepted"]:
                    raise ValueError(
                        "Fixed demonstration failed: "
                        + "; ".join(result["verification"]["errors"])
                    )
                examples.append(
                    {
                        "sample_id": row["sample_id"],
                        "output": result["output"],
                        "text": example_text(index, row, result),
                        "verification": result["verification"],
                    }
                )
            return usage, examples, assemble(frame, bank, usage, examples)
        except (ValueError, TypeError, KeyError) as exc:
            feedback = [str(exc)]
            write_json(folder / f"rejected_{attempt}.json", {"errors": feedback})
    raise ValueError("Unable to synchronize and verify the initial prompt")


def create_seed(
    backend: Any,
    args: argparse.Namespace,
    demos: list[dict],
    frame: str,
    templates: dict,
) -> dict:
    feedback: list[str] = []
    folder = args.output_dir / "initialization"
    for attempt in range(args.repair_attempts):
        value = backend.request(
            folder / f"bank_{attempt}.json",
            model=args.teacher_model,
            instruction=templates["create_skills"],
            payload={"domain": args.domain, "fixed_frame": frame, "feedback": feedback},
            images=[],
        )
        try:
            bank = validate_bank(value, initial=True)
            bank["domain"] = args.domain
            usage, examples, prompt = synchronize(
                backend, args, bank, demos, frame, templates, folder / f"sync_{attempt}"
            )
            version = args.output_dir / "versions" / "v0000"
            write_json(version / "skills.json", bank)
            write_json(version / "examples.json", examples)
            write_text(version / "how_to_use.txt", usage)
            write_text(version / "full_prompt.txt", prompt)
            state = {
                "version": 0,
                "version_path": "versions/v0000",
                "next_window": 0,
                "status": "seeded",
            }
            write_json(args.output_dir / "state.json", state)
            return state
        except ValueError as exc:
            feedback = [str(exc)]
            write_json(folder / f"rejected_{attempt}.json", {"errors": feedback})
    raise ValueError("Seed creation failed; no prompt was committed")


def run(args: argparse.Namespace, backend: Any = None) -> dict:
    _, demos, frame, templates, manifest = prepare(args)
    manifest["backend"] = "api" if backend is None else type(backend).__name__
    if args.dry_run:
        summary = {
            key: value
            for key, value in manifest.items()
            if key not in ("train_ids", "validation_ids", "test_ids")
        }
        summary.update(
            {f"{name}_count": len(manifest[f"{name}_ids"]) for name in ("train", "validation", "test")}
        )
        print(json.dumps({"dry_run": True, **summary}, ensure_ascii=False, indent=2))
        return manifest

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise ValueError("Seed inputs/configuration changed; use a new output directory")
        write_json(manifest_path, manifest)
        state_path = args.output_dir / "state.json"
        if state_path.exists():
            state = read_json(state_path)
            version = args.output_dir / state["version_path"]
            prompt_path = version / "full_prompt.txt"
            if state.get("status") != "seeded" or not prompt_path.is_file():
                raise ValueError("Existing seed state is incomplete or inconsistent")
            return state
        backend = backend or APIBackend(
            args.timeout, args.max_tokens, image_max_pixels=args.image_max_pixels
        )
        return create_seed(backend, args, demos, frame, templates)


def main() -> None:
    args = parse_args()
    try:
        state = run(args)
    except (ValueError, RuntimeError, FileNotFoundError, BlockingIOError) as exc:
        raise SystemExit(str(exc)) from None
    if not args.dry_run:
        print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
