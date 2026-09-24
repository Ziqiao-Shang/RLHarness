"""Deterministic release checks that do not require models, GPUs, or API calls."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from rlharness.common.config import ROOT


DOMAINS = ("metromap", "travelmap")
PROMPT_SECTIONS = (
    "# Planning Skills",
    "# How to Use the Skills",
    "# Required Procedure",
    "# Few-Shot Demonstrations",
    "# Output Format Constraints",
    "[Relevant Strategy Selection]",
    "<response>",
)
REQUIRED_PATHS = (
    "README.md",
    "NOTICE.md",
    "docs/environment.md",
    "pyproject.toml",
    "requirements.txt",
    ".env.example",
    "scripts/verify_release.sh",
    "configs/README.md",
    "configs/skillopt.yaml",
    "configs/sft_train.yaml",
    "configs/sft_merge.yaml",
    "configs/rl_round1.yaml",
    "configs/rl_round2.yaml",
    "prompts/data_process/teacher_metromap.txt",
    "prompts/data_process/teacher_travelmap.txt",
    "prompts/evolution/metromap/select.txt",
    "prompts/evolution/travelmap/select.txt",
    "src/skillopt/engine/trainer.py",
    "src/skillopt/prompts/analyst_error.md",
    "third_party/SkillOpt/LICENSE",
    "third_party/SkillOpt/README.md",
    "third_party/README.md",
    "src/rlharness/evolution/skill_select.py",
)

FORBIDDEN_LEGACY_PATHS = (
    "scripts/select_candidate.sh",
    "src/rlharness/initialization/pipeline.py",
    "src/rlharness/initialization/handoff.py",
    "prompts/initialization/metromap/update_skills.txt",
    "prompts/initialization/travelmap/update_skills.txt",
)
TEXT_SUFFIXES = {
    ".cfg", ".env", ".ini", ".json", ".md", ".py", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_yaml_chain(path: Path, errors: list[str], seen: set[Path] | None = None) -> None:
    seen = set() if seen is None else seen
    resolved = path.resolve()
    if resolved in seen:
        errors.append(f"cyclic config inheritance: {path}")
        return
    if not path.is_file():
        errors.append(f"missing config: {path}")
        return
    seen.add(resolved)
    match = re.search(r"(?m)^_base_config:\s*([^#\n]+)", path.read_text(encoding="utf-8"))
    if match:
        value = match.group(1).strip().strip("'\"")
        _check_yaml_chain(path.parent / value, errors, seen)


def _scan_credentials(root: Path, errors: list[str]) -> None:
    token = re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")
    assignment = re.compile(
        rb"(?m)^[ \t]*(?:export[ \t]+)?(?:OPENLUX_API_KEY|WANDB_API_KEY|API_KEY)"
        rb"[ \t]*=[ \t]*[\"']?([^ \t\r\n\"']*)"
    )
    placeholders = (b"YOUR", b"REPLACE", b"EXAMPLE", b"PLACEHOLDER", b"CHANGEME")
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        if "third_party" in path.parts and not path.name.startswith(".env"):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and not path.name.startswith(".env"):
            continue
        data = path.read_bytes()
        if token.search(data):
            errors.append(f"credential-like token in {path.relative_to(root)}")
        for match in assignment.finditer(data):
            value = match.group(1).strip()
            upper = value.upper()
            if value and not value.startswith(
                (b"$", b"<", b"{", b"os.environ", b"getenv(", b"str(", b"None")
            ) and not any(
                item in upper for item in placeholders
            ):
                errors.append(f"literal API key assignment in {path.relative_to(root)}")


def verify_release(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            errors.append(f"missing required path: {relative}")

    for relative in FORBIDDEN_LEGACY_PATHS:
        if (root / relative).exists():
            errors.append(f"obsolete public entry or initialization path is present: {relative}")

    for domain in DOMAINS:
        for name in ("original.txt", "final.txt", "original_skills.json", "frame.txt"):
            path = root / "prompts" / "student" / domain / name
            if not path.is_file():
                errors.append(f"missing domain prompt: {path.relative_to(root)}")
        split_dir = root / "data" / "reference" / domain / "splits"
        splits: dict[str, list[str]] = {}
        for name, expected in (("train", 1600), ("validation", 100), ("test", 400)):
            path = split_dir / f"{name}_sample_ids.json"
            if not path.is_file():
                errors.append(f"missing split IDs: {path.relative_to(root)}")
                continue
            values = _json(path)
            if len(values) != expected or len(set(values)) != expected:
                errors.append(f"invalid {domain}/{name} split size or duplicates")
            splits[name] = values
        if len(splits) == 3:
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                if set(splits[left]) & set(splits[right]):
                    errors.append(f"overlapping {domain} splits: {left}/{right}")

        for name in ("original.txt", "final.txt"):
            path = root / "prompts" / "student" / domain / name
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                missing = [section for section in PROMPT_SECTIONS if section not in text]
                if missing:
                    errors.append(f"{path.relative_to(root)} missing sections: {missing}")

    for round_name in ("rl_round1.yaml", "rl_round2.yaml"):
        _check_yaml_chain(root / "configs" / round_name, errors)

    fixed_split_scripts = (
        "01_generate_sft_data.sh",
        "03_train_rl_round1.sh",
        "04_evolve_skills.sh",
        "05_train_rl_round2.sh",
        "06_evaluate_test.sh",
    )
    for name in fixed_split_scripts:
        path = root / "scripts" / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        required = (
            "--fixed-ids-dir",
            "$ROOT/data/reference/metromap/splits",
            "$ROOT/data/reference/travelmap/splits",
        )
        if any(value not in text for value in required) or "FIXED_SPLIT_IDS_DIR" in text:
            errors.append(f"formal script can bypass locked split IDs: scripts/{name}")

    sft_script = root / "scripts" / "02_train_sft.sh"
    if sft_script.is_file():
        text = sft_script.read_text(encoding="utf-8")
        if "third_party/LlamaFactory/data/dataset_info.json" in text:
            errors.append("SFT script mutates the vendored LLaMA-Factory registry")
        for required in ("REGISTRY_DIR", "dataset_dir=$REGISTRY_DIR"):
            if required not in text:
                errors.append(f"SFT script is missing isolated registry setting: {required}")

    sft_data_script = root / "scripts" / "01_generate_sft_data.sh"
    if sft_data_script.is_file():
        text = sft_data_script.read_text(encoding="utf-8")
        if 'MODEL="${TEACHER_MODEL:-gpt-5.6-sol}"' not in text:
            errors.append("SFT teacher launcher does not default to gpt-5.6-sol")

    eval_script = root / "scripts" / "06_evaluate_test.sh"
    if eval_script.is_file():
        text = eval_script.read_text(encoding="utf-8")
        required_defaults = (
            'DEFAULT_MODEL="$ROOT/artifacts/sft_merged"',
            'DEFAULT_ROUND2_OUTPUT="$ROOT/artifacts/grpo_round2"',
            'DEFAULT_MODEL="$ROOT/artifacts/travelmap/sft_merged"',
            'DEFAULT_ROUND2_OUTPUT="$ROOT/artifacts/travelmap/grpo_round2"',
            'PROMPT="$ROUND2_OUTPUT/selected_prompt.txt"',
            "verify-round2",
        )
        for required in required_defaults:
            if required not in text:
                errors.append(f"test launcher does not follow round-two output: {required}")
        if '$ROOT/../metromap_final' in text or '$ROOT/../travelmap_final' in text:
            errors.append("test launcher defaults to an external published model")

    round2_script = root / "scripts" / "05_train_rl_round2.sh"
    if round2_script.is_file():
        text = round2_script.read_text(encoding="utf-8")
        required_selection_contract = (
            'SELECTION_DIR="${SELECTED_SKILL_DIR:-$DEFAULT_SELECTION}"',
            'PROMPT="$SELECTION_DIR/full_prompt.txt"',
            "lock-round2",
            'PROMPT="$OUTPUT/selected_prompt.txt"',
        )
        for required in required_selection_contract:
            if required not in text:
                errors.append(f"round-two launcher is missing Prompt lock: {required}")

    round1_script = root / "scripts" / "03_train_rl_round1.sh"
    if round1_script.is_file():
        text = round1_script.read_text(encoding="utf-8")
        for required in ("lock-round1", 'PROMPT="$OUTPUT/selected_prompt.txt"'):
            if required not in text:
                errors.append(f"round-one launcher is missing Prompt lock: {required}")

    for package in ("rlharness", "skillopt"):
        for path in (root / "src" / package).rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"Python syntax error in {path.relative_to(root)}: {exc}")

    forbidden_dirs = (root / "models", root / "data" / "raw")
    for path in forbidden_dirs:
        if path.exists():
            errors.append(f"release contains external assets: {path.relative_to(root)}")
    cache_paths = list(root.rglob("__pycache__")) + list(root.rglob(".pytest_cache"))
    if cache_paths:
        warnings.append(f"generated cache directories present: {len(cache_paths)}")

    legacy_package = "metromap" + "_skill_rl"
    for path in root.rglob("*"):
        if not path.is_file() or "third_party" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if legacy_package in path.read_text(encoding="utf-8", errors="ignore"):
            errors.append(f"legacy package reference in {path.relative_to(root)}")

    _scan_credentials(root, errors)
    return {
        "root": str(root),
        "ok": not errors,
        "domains": list(DOMAINS),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = verify_release(args.root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for warning in report["warnings"]:
            print(f"WARNING {warning}")
        for error in report["errors"]:
            print(f"ERROR   {error}")
        print("RELEASE PASS" if report["ok"] else "RELEASE FAIL")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
