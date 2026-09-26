"""Prepare the exact MapTab records referenced by RLHarness split IDs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAPTAB_ROOT = Path(os.environ.get("MAPTAB_ROOT", ROOT.parent / "maptab_data"))
PUBLIC_DATASET_REPO = "szq-nju/MapTab"
# This complete official release has the raw row order that defines locked IDs.
PUBLIC_REVISION = os.environ.get(
    "RLHARNESS_MAPTAB_REVISION",
    "83efbbee2d6f2f8a1fbf278201f3754029b341f1",
)

SOURCE_FILES = {
    domain: {
        split: (
            f"raw/{domain}/data/{'training_set' if split == 'train' else 'test_set'}/"
            f"{domain}_shortest_path_query_map_and_tab_with_constraint_1_2_3_4"
            f"_only_vertex2_{'training' if split == 'train' else 'test'}_set.json"
        )
        for split in ("train", "test")
    }
    for domain in ("metromap", "travelmap")
}
PROMPT_FILES = {
    domain: (
        f"raw/{domain}/prompts/"
        f"{domain}_shortest_path_with_constraint_1_2_3_4_only_vertex2.txt"
    )
    for domain in ("metromap", "travelmap")
}
EXPECTED_ROWS = {
    "metromap": {"train": 6400, "test": 1600},
    "travelmap": {"train": 6720, "test": 1680},
}
ASSET_FIELDS = ("figure", "vertex_tab", "vertex2_tab", "edge_tab")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def selected_domains(domain: str) -> tuple[str, ...]:
    return ("metromap", "travelmap") if domain == "all" else (domain,)


def resolve_source(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = (
        path,
        path / "MapTab",
        path / "data" / "raw" / "MapTab",
        path / "dataset" / "MapTab",
    )
    for candidate in candidates:
        if (candidate / "metromap").is_dir() or (candidate / "travelmap").is_dir():
            return candidate
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not locate a MapTab root. Searched:\n  {searched}")


def copy_domains(source: Path, destination: Path, domain: str) -> None:
    domains = selected_domains(domain)
    for name in domains:
        source_dir = source / name
        destination_dir = destination / name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source domain directory: {source_dir}")
        if destination_dir.exists():
            raise RuntimeError(
                f"Destination already exists: {destination_dir}. "
                "Use --check-only to inspect it or choose a new destination."
            )
    destination.mkdir(parents=True, exist_ok=True)
    for name in domains:
        shutil.copytree(source / name, destination / name)
        print(f"[copy] {name}: {source / name} -> {destination / name}")


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return rows


def _local_path_for_remote(root: Path, remote_path: str) -> Path:
    relative = PurePosixPath(remote_path)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 3:
        raise ValueError(f"Unsafe MapTab repository path: {remote_path}")
    if relative.parts[0] not in {"raw", "assets"}:
        raise ValueError(f"Unsupported MapTab repository path: {remote_path}")
    return root.joinpath(*relative.parts[1:])


def _split_rows(root: Path, domain: str, split: str) -> list[dict[str, Any]]:
    direct = _local_path_for_remote(root, SOURCE_FILES[domain][split])
    if direct.is_file():
        return load_json_rows(direct)

    combined = (
        root
        / domain
        / "data"
        / "all"
        / f"{domain}_shortest_path_query_map_and_tab_with_constraint_1_2_3_4"
        "_only_vertex2.json"
    )
    if combined.is_file():
        category = "training_set" if split == "train" else "test_set"
        return [
            row
            for row in load_json_rows(combined)
            if row.get("set_category") == category
        ]
    raise FileNotFoundError(direct)


def verify_dataset_shape(root: Path, domain: str) -> None:
    for current_domain in selected_domains(domain):
        counts = {
            split: len(_split_rows(root, current_domain, split))
            for split in ("train", "test")
        }
        if counts != EXPECTED_ROWS[current_domain]:
            raise RuntimeError(
                f"Unexpected {current_domain} row counts: "
                f"{counts['train']}/{counts['test']}"
            )
        print(
            f"[shape] {current_domain} train/test="
            f"{counts['train']}/{counts['test']}"
        )


def _split_root(domain: str) -> Path:
    return ROOT / "data" / "reference" / domain / "splits"


def locked_sample_ids(domain: str, *, include_prompt_examples: bool = True) -> set[str]:
    split_root = _split_root(domain)
    sample_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        values = json.loads(
            (split_root / f"{split}_sample_ids.json").read_text(encoding="utf-8")
        )
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise RuntimeError(
                f"Invalid locked ID file: {split_root / f'{split}_sample_ids.json'}"
            )
        sample_ids.update(values)
    if include_prompt_examples:
        manifest = json.loads(
            (split_root / "manifest.json").read_text(encoding="utf-8")
        )
        sample_ids.update(manifest.get("prompt_demonstration_sample_ids", []))
    return sample_ids


def parse_sample_id(sample_id: str, domain: str) -> tuple[str, int]:
    parts = sample_id.split(":")
    if len(parts) != 3 or parts[0] != domain or parts[1] not in {"train", "test"}:
        raise ValueError(f"Invalid {domain} sample ID: {sample_id}")
    if not parts[2].isdigit():
        raise ValueError(f"Invalid sample index: {sample_id}")
    index = int(parts[2])
    if not 0 <= index < EXPECTED_ROWS[domain][parts[1]]:
        raise ValueError(f"Sample ID out of range: {sample_id}")
    return parts[1], index


def verify_locked_splits(domain: str) -> None:
    expected_sizes = {"train": 1600, "validation": 100, "test": 400}
    source_split = {"train": "train", "validation": "train", "test": "test"}
    for current_domain in selected_domains(domain):
        split_sets: dict[str, set[str]] = {}
        split_root = _split_root(current_domain)
        for split, expected_size in expected_sizes.items():
            ids = json.loads(
                (split_root / f"{split}_sample_ids.json").read_text(encoding="utf-8")
            )
            if len(ids) != expected_size or len(set(ids)) != expected_size:
                raise RuntimeError(
                    f"Invalid {current_domain}/{split} ID count or duplicates"
                )
            for sample_id in ids:
                parsed_split, _ = parse_sample_id(sample_id, current_domain)
                if parsed_split != source_split[split]:
                    raise RuntimeError(f"Invalid sample ID source split: {sample_id}")
            split_sets[split] = set(ids)
        if split_sets["train"] & split_sets["validation"]:
            raise RuntimeError(f"{current_domain} train and validation IDs overlap")
        if split_sets["test"] & (split_sets["train"] | split_sets["validation"]):
            raise RuntimeError(f"{current_domain} test IDs overlap with train IDs")
        print(f"[splits] {current_domain} locked IDs=1600/100/400")


def _normalize_asset_reference(domain: str, value: str) -> tuple[str, Path]:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe MapTab asset path: {value}")
    parts = (
        relative.parts[1:]
        if relative.parts and relative.parts[0] == "assets"
        else relative.parts
    )
    if len(parts) < 3 or parts[0] != domain:
        raise ValueError(f"Asset does not belong to {domain}: {value}")
    return PurePosixPath("assets", *parts).as_posix(), Path(*parts)


def required_assets_for_ids(
    domain: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    sample_ids: Iterable[str],
) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for sample_id in sorted(set(sample_ids)):
        split, index = parse_sample_id(sample_id, domain)
        try:
            row = rows_by_split[split][index]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Official source does not contain {sample_id}") from exc
        for field in ASSET_FIELDS:
            value = row.get(field)
            if not value:
                continue
            remote, local = _normalize_asset_reference(domain, str(value))
            assets[remote] = local
    return assets


def verify_locked_assets(root: Path, domain: str) -> None:
    for current_domain in selected_domains(domain):
        rows = {
            split: _split_rows(root, current_domain, split)
            for split in ("train", "test")
        }
        assets = required_assets_for_ids(
            current_domain,
            rows,
            locked_sample_ids(current_domain),
        )
        missing = [
            str(path) for path in assets.values() if not (root / path).is_file()
        ]
        if missing:
            preview = "\n  ".join(missing[:10])
            raise FileNotFoundError(
                f"Missing {len(missing)} assets for locked {current_domain} IDs:"
                f"\n  {preview}"
            )
        print(f"[assets] {current_domain} required files={len(assets)}")


def _materialize_file(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source.resolve(), temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_path(destination: Path, domain: str) -> Path:
    return destination / ".rlharness" / f"{domain}.json"


def _manifest_is_complete(
    destination: Path,
    domain: str,
    repo_id: str,
    revision: str,
) -> bool:
    path = _manifest_path(destination, domain)
    if not path.is_file() or not SHA_PATTERN.fullmatch(revision):
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("repo_id") != repo_id
        or manifest.get("resolved_revision") != revision
    ):
        return False
    files = manifest.get("materialized_files")
    return bool(files) and all(
        (destination / relative).is_file() for relative in files
    )


def _resolved_revision(repo_id: str, revision: str) -> str:
    if SHA_PATTERN.fullmatch(revision):
        return revision
    info = HfApi().dataset_info(repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve {repo_id}@{revision}")
    return info.sha


def _download_source_rows(
    domain: str,
    *,
    repo_id: str,
    revision: str,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, remote_path in SOURCE_FILES[domain].items():
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=remote_path,
                revision=revision,
            )
        )
        rows_by_split[split] = load_json_rows(cached)
    actual = {split: len(rows) for split, rows in rows_by_split.items()}
    if actual != EXPECTED_ROWS[domain]:
        raise RuntimeError(
            f"Official {domain}@{revision} row counts changed: {actual}; "
            f"expected {EXPECTED_ROWS[domain]}. Locked IDs cannot be applied safely."
        )
    return rows_by_split


def _smoke_sample_ids(domain: str, count_per_split: int) -> list[str]:
    locked = locked_sample_ids(domain, include_prompt_examples=False)
    selected: list[str] = []
    for split in ("train", "test"):
        candidates = sorted(
            sample_id
            for sample_id in locked
            if sample_id.startswith(f"{domain}:{split}:")
        )
        selected.extend(candidates[:count_per_split])
    return selected


def smoke_test_official(
    domain: str,
    *,
    repo_id: str = PUBLIC_DATASET_REPO,
    revision: str = PUBLIC_REVISION,
    count_per_split: int = 1,
) -> list[dict[str, Any]]:
    """Download a few locked-ID records and verify their referenced assets."""
    if count_per_split < 1:
        raise ValueError("count_per_split must be positive")
    resolved_revision = _resolved_revision(repo_id, revision)
    results: list[dict[str, Any]] = []
    for current_domain in selected_domains(domain):
        rows_by_split = _download_source_rows(
            current_domain,
            repo_id=repo_id,
            revision=resolved_revision,
        )
        for sample_id in _smoke_sample_ids(current_domain, count_per_split):
            split, index = parse_sample_id(sample_id, current_domain)
            row = rows_by_split[split][index]
            assets = required_assets_for_ids(
                current_domain,
                rows_by_split,
                [sample_id],
            )
            checked_assets: list[dict[str, Any]] = []
            for remote_path in sorted(assets):
                cached = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=remote_path,
                        revision=resolved_revision,
                    )
                )
                if "/images/" in remote_path:
                    with Image.open(cached) as image:
                        image.verify()
                    asset_type = "image"
                elif cached.suffix == ".json":
                    with cached.open(encoding="utf-8") as handle:
                        json.load(handle)
                    asset_type = "json"
                else:
                    if not cached.read_bytes():
                        raise RuntimeError(f"Downloaded empty asset: {remote_path}")
                    asset_type = "file"
                checked_assets.append(
                    {
                        "type": asset_type,
                        "remote_path": remote_path,
                        "bytes": cached.stat().st_size,
                    }
                )
            result = {
                "sample_id": sample_id,
                "question": row.get("question"),
                "route": (row.get("routes") or [None])[0],
                "assets": checked_assets,
            }
            results.append(result)
            print(
                f"[smoke] {sample_id}: assets={len(checked_assets)} "
                f"question={str(row.get('question', ''))[:80]!r}"
            )
    print(
        f"Official MapTab API smoke test passed: samples={len(results)}, "
        f"revision={resolved_revision}"
    )
    return results


def _download_domain(
    destination: Path,
    domain: str,
    *,
    repo_id: str,
    revision: str,
    max_workers: int,
) -> None:
    if _manifest_is_complete(destination, domain, repo_id, revision):
        print(f"[download] reuse complete {domain} data at {destination}")
        verify_dataset_shape(destination, domain)
        verify_locked_assets(destination, domain)
        return

    resolved_revision = _resolved_revision(repo_id, revision)
    if _manifest_is_complete(destination, domain, repo_id, resolved_revision):
        print(f"[download] reuse complete {domain} data at {destination}")
        verify_dataset_shape(destination, domain)
        verify_locked_assets(destination, domain)
        return

    rows_by_split = _download_source_rows(
        domain,
        repo_id=repo_id,
        revision=resolved_revision,
    )

    sample_ids = locked_sample_ids(domain)
    assets = required_assets_for_ids(domain, rows_by_split, sample_ids)
    remote_files = set(assets)
    remote_files.update(SOURCE_FILES[domain].values())
    remote_files.add(PROMPT_FILES[domain])
    print(
        f"[download] {domain}: {len(sample_ids)} IDs, {len(assets)} unique assets, "
        f"revision={resolved_revision}"
    )
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            allow_patterns=sorted(remote_files),
            max_workers=max_workers,
        )
    )

    materialized: list[str] = []
    for remote_path in sorted(remote_files):
        source = snapshot / remote_path
        if not source.is_file():
            raise FileNotFoundError(f"Hub snapshot is missing {remote_path}")
        local = _local_path_for_remote(destination, remote_path)
        _materialize_file(source, local)
        materialized.append(local.relative_to(destination).as_posix())

    manifest = {
        "source": f"https://huggingface.co/datasets/{repo_id}",
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "domain": domain,
        "selection": (
            "locked_train1600_validation100_test400_plus_prompt_examples"
        ),
        "locked_split_ids": 2100,
        "prompt_example_ids": len(sample_ids) - 2100,
        "selected_sample_ids": len(sample_ids),
        "unique_assets": len(assets),
        "source_files": SOURCE_FILES[domain],
        "materialized_files": materialized,
    }
    manifest_path = _manifest_path(destination, domain)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verify_dataset_shape(destination, domain)
    verify_locked_assets(destination, domain)


def download_official_locked(
    destination: Path,
    domain: str,
    *,
    repo_id: str = PUBLIC_DATASET_REPO,
    revision: str = PUBLIC_REVISION,
    max_workers: int = 8,
    dry_run: bool = False,
) -> None:
    destination = destination.expanduser().resolve()
    if dry_run:
        print(f"Repository: https://huggingface.co/datasets/{repo_id}")
        print(f"Revision: {revision}")
        print(f"Destination: {destination}")
        for current_domain in selected_domains(domain):
            print(
                f"{current_domain}: {len(locked_sample_ids(current_domain))} "
                f"selected IDs; metadata="
                f"{','.join(SOURCE_FILES[current_domain].values())}"
            )
        print("Dry run only; assets are derived from those source rows by sample ID.")
        return

    destination.mkdir(parents=True, exist_ok=True)
    verify_locked_splits(domain)
    for current_domain in selected_domains(domain):
        _download_domain(
            destination,
            current_domain,
            repo_id=repo_id,
            revision=revision,
            max_workers=max_workers,
        )
    print(f"Official MapTab data ready: {destination}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official MapTab rows/assets referenced by locked RLHarness "
            "IDs, import a local snapshot, or verify an existing data root."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--source",
        type=Path,
        help="Existing MapTab snapshot or parent directory",
    )
    mode.add_argument(
        "--download",
        "--download-locked",
        "--download-public-test",
        dest="download",
        action="store_true",
        help=(
            "Download the locked-ID subset from the complete official "
            "Hugging Face release"
        ),
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Download one or more locked train/test examples per domain and "
            "validate their image/table assets without materializing the dataset"
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="Output root; defaults to ../maptab_data",
    )
    parser.add_argument(
        "--domain",
        choices=("all", "metromap", "travelmap"),
        default="all",
    )
    parser.add_argument("--repo-id", default=PUBLIC_DATASET_REPO)
    parser.add_argument("--revision", default=PUBLIC_REVISION)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--smoke-count",
        type=int,
        default=1,
        help="Locked examples to test from each source split (default: 1)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify source or destination without copying",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the official download without downloading",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    if args.smoke_count < 1:
        raise SystemExit("--smoke-count must be positive")
    if args.dry_run and not args.download:
        raise SystemExit("--dry-run is only valid with --download")

    destination = (args.destination or DEFAULT_MAPTAB_ROOT).expanduser().resolve()
    if args.smoke_test:
        smoke_test_official(
            args.domain,
            repo_id=args.repo_id,
            revision=args.revision,
            count_per_split=args.smoke_count,
        )
        return
    if args.download:
        download_official_locked(
            destination,
            args.domain,
            repo_id=args.repo_id,
            revision=args.revision,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
        )
        return

    if args.source:
        source = resolve_source(args.source)
        verify_dataset_shape(source, args.domain)
        verify_locked_splits(args.domain)
        verify_locked_assets(source, args.domain)
        if args.check_only:
            print(f"MapTab source is compatible: {source}")
            return
        if source == destination:
            raise SystemExit("Source and destination are identical; use --check-only instead.")
        copy_domains(source, destination, args.domain)
    elif not args.check_only:
        raise SystemExit("Choose --download, --source PATH, or --check-only.")

    verify_dataset_shape(destination, args.domain)
    verify_locked_splits(args.domain)
    verify_locked_assets(destination, args.domain)
    print(f"MapTab data ready: {destination}")


if __name__ == "__main__":
    main()
