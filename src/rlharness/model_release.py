#!/usr/bin/env python3
"""Download and validate one released RLHarness model from the Hub."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from rlharness.common.config import ROOT


DEFAULT_MODEL_REPO = "szq-nju/RLHarness-MapTab-Models"
DEFAULT_MODEL_SUBFOLDERS = {
    "metromap": "metromap_trained",
    "travelmap": "travelmap_trained",
}
REQUIRED_MODEL_FILES = ("config.json", "model.safetensors.index.json")


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
    parser.add_argument(
        "--subfolder",
        default=os.environ.get("RLHARNESS_MODEL_SUBFOLDER"),
        help=(
            "Model subfolder in the Hub repository. Defaults to the published "
            "domain mapping; RLHARNESS_MODEL_SUBFOLDER provides the same override."
        ),
    )
    parser.add_argument(
        "--print-local-path",
        action="store_true",
        help="Print only the validated local model directory for shell integration.",
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=(
            Path(os.environ["RLHARNESS_LOCAL_MODEL_PATH"])
            if os.environ.get("RLHARNESS_LOCAL_MODEL_PATH")
            else None
        ),
        help=(
            "Validate and use an existing fully merged checkpoint instead of "
            "downloading from the Hub."
        ),
    )
    return parser.parse_args()


def _normalise_subfolder(domain: str, subfolder: str | None) -> str:
    raw = subfolder or DEFAULT_MODEL_SUBFOLDERS[domain]
    selected = raw.strip("/")
    parts = Path(selected).parts
    if (
        not selected
        or Path(raw).is_absolute()
        or selected == "."
        or ".." in parts
    ):
        raise ValueError(f"Invalid model subfolder: {selected!r}")
    return selected


def _validate_model_dir(model_dir: Path) -> list[str]:
    model_dir = model_dir.expanduser().resolve()
    for name in REQUIRED_MODEL_FILES:
        if not (model_dir / name).is_file():
            raise SystemExit(f"Model is incomplete: {model_dir / name}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("model_type"):
        raise SystemExit(f"Model config has no model_type: {model_dir / 'config.json'}")

    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(
            f"Model index has no weight_map: "
            f"{model_dir / 'model.safetensors.index.json'}"
        )
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        relative = Path(str(shard))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"Unsafe model shard path in index: {shard}")
        path = model_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Model shard is missing or empty: {path}")
    return [str(shard) for shard in shards]


def validate_local_model(*, domain: str, model_path: Path) -> dict[str, Any]:
    """Validate an existing merged checkpoint without any network access."""
    if domain not in DEFAULT_MODEL_SUBFOLDERS:
        raise ValueError(f"Unsupported domain: {domain}")
    model_dir = model_path.expanduser().resolve()
    shards = _validate_model_dir(model_dir)
    return {
        "domain": domain,
        "model": {
            "source": "local",
            "local_path": str(model_dir),
            "weight_shards": shards,
        },
    }


def download_released_model(
    *,
    domain: str,
    output_dir: Path,
    model_repo_id: str = DEFAULT_MODEL_REPO,
    revision: str = "main",
    subfolder: str | None = None,
) -> dict[str, Any]:
    """Download a complete domain checkpoint and return its resolved coordinates."""
    if domain not in DEFAULT_MODEL_SUBFOLDERS:
        raise ValueError(f"Unsupported domain: {domain}")

    selected_subfolder = _normalise_subfolder(domain, subfolder)
    root = output_dir.expanduser().resolve()
    model_dir = root / selected_subfolder
    root.mkdir(parents=True, exist_ok=True)

    model_info = HfApi().model_info(model_repo_id, revision=revision)
    remote_files = {
        sibling.rfilename
        for sibling in (model_info.siblings or [])
        if getattr(sibling, "rfilename", None)
    }
    missing_remote = [
        f"{selected_subfolder}/{name}"
        for name in REQUIRED_MODEL_FILES
        if f"{selected_subfolder}/{name}" not in remote_files
    ]
    if missing_remote:
        missing = ", ".join(missing_remote)
        raise SystemExit(
            f"Hub model subfolder {selected_subfolder!r} is incomplete; missing: {missing}"
        )

    snapshot_download(
        repo_id=model_repo_id,
        repo_type="model",
        revision=revision,
        local_dir=root,
        allow_patterns=[f"{selected_subfolder}/*"],
    )

    shards = _validate_model_dir(model_dir)

    resolution = {
        "domain": domain,
        "model": {
            "repo_id": model_repo_id,
            "requested_revision": revision,
            "resolved_commit": model_info.sha,
            "subfolder": selected_subfolder,
            "local_path": str(model_dir),
            "weight_shards": shards,
        },
    }
    (root / f"{domain}_resolution.json").write_text(
        json.dumps(resolution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolution


def main() -> None:
    args = parse_args()
    if args.local_path:
        resolution = validate_local_model(
            domain=args.domain,
            model_path=args.local_path,
        )
    else:
        resolution = download_released_model(
            domain=args.domain,
            output_dir=args.output_dir,
            model_repo_id=args.model_repo_id,
            revision=args.revision,
            subfolder=args.subfolder,
        )
    if args.print_local_path:
        print(resolution["model"]["local_path"])
    else:
        print(json.dumps(resolution, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
