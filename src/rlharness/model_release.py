#!/usr/bin/env python3
"""Download one released RLHarness model from the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from rlharness.common.config import ROOT


DEFAULT_MODEL_REPO = "szq-nju/RLHarness-MapTab-Models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("metromap", "travelmap"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("RLHARNESS_MODEL_ROOT", ROOT / "models" / "release")),
    )
    parser.add_argument(
        "--model-repo-id",
        default=os.environ.get("RLHARNESS_MODEL_REPO", DEFAULT_MODEL_REPO),
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("RLHARNESS_MODEL_REVISION", "main"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    model_dir = root / args.domain
    root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model_repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=root,
        allow_patterns=[f"{args.domain}/*", "manifest.json"],
    )

    for name in ("config.json", "model.safetensors.index.json"):
        if not (model_dir / name).is_file():
            raise SystemExit(f"Downloaded model is incomplete: {model_dir / name}")

    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    for shard in sorted(set(index.get("weight_map", {}).values())):
        if not (model_dir / shard).is_file():
            raise SystemExit(f"Downloaded model shard is missing: {model_dir / shard}")

    model_info = HfApi().model_info(args.model_repo_id, revision=args.revision)
    resolution = {
        "domain": args.domain,
        "model": {
            "repo_id": args.model_repo_id,
            "requested_revision": args.revision,
            "resolved_commit": model_info.sha,
            "subfolder": args.domain,
            "local_path": str(model_dir),
        },
    }
    (root / f"{args.domain}_resolution.json").write_text(
        json.dumps(resolution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(resolution, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
