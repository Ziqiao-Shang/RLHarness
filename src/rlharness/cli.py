"""Thin command dispatcher for the public reproduction stages."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from rlharness.common.config import ROOT


STAGES = {
    "prepare-data": "prepare_data.sh",
    "initialize": "00_initialize_skills.sh",
    "sft-data": "01_generate_sft_data.sh",
    "sft-train": "02_train_sft.sh",
    "rl-round1": "03_train_rl_round1.sh",
    "evolve": "04_evolve_skills.sh",
    "rl-round2": "05_train_rl_round2.sh",
    "evaluate": "06_evaluate_test.sh",
}


def command_for(stage: str, domain: str, extra: list[str] | None = None) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    command = ["bash", str(ROOT / "scripts" / STAGES[stage]), "--domain", domain]
    return command + list(extra or [])


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run one reproducible RLHarness stage."
    )
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--domain", choices=("metromap", "travelmap"), default="metromap")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved shell command without running it.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the release and print the stage command without running it.",
    )
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def main(argv: list[str] | None = None) -> int:
    args, extra = parse_args(argv)
    command = command_for(args.stage, args.domain, extra)
    if args.check_only:
        from rlharness.release import verify_release

        report = verify_release(ROOT)
        if report["errors"]:
            for error in report["errors"]:
                print(f"ERROR: {error}")
            return 1
    if args.print_command or args.check_only:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
