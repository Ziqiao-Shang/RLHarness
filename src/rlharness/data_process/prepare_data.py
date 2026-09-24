"""Prepare external MapTab data without committing raw assets to the repository."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAPTAB_ROOT = Path(os.environ.get("MAPTAB_ROOT", ROOT.parent / "maptab_data"))
PUBLIC_REVISION = "93bf27ae4387ff33f61d627063c77c9596e3e67c"
PUBLIC_BASE_URL = f"https://huggingface.co/datasets/szq-nju/MapTab/resolve/{PUBLIC_REVISION}"


@dataclass(frozen=True)
class PublicArchive:
    domain: str
    name: str


PUBLIC_ARCHIVES = (
    PublicArchive("metromap", "metromap_prompts.zip"),
    PublicArchive("metromap", "metromap_rp_images_test_set.zip"),
    PublicArchive("metromap", "metromap_rp_queries_test_set.zip"),
    PublicArchive("metromap", "metromap_rp_tabulars_test_set.zip"),
    PublicArchive("travelmap", "travelmap_prompts.zip"),
    PublicArchive("travelmap", "travelmap_rp_images_test_set.zip"),
    PublicArchive("travelmap", "travelmap_rp_queries_test_set.zip"),
    PublicArchive("travelmap", "travelmap_rp_tabulars_test_set.zip"),
)


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
    domains = ("metromap", "travelmap") if domain == "all" else (domain,)
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


def load_json_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return rows


def verify_dataset_shape(root: Path, domain: str) -> None:
    if domain in ("all", "metromap"):
        base = root / "metromap" / "data"
        train = load_json_rows(base / "training_set" / "metromap_shortest_path_query_map_and_tab_with_constraint_1_2_3_4_only_vertex2_training_set.json")
        test = load_json_rows(base / "test_set" / "metromap_shortest_path_query_map_and_tab_with_constraint_1_2_3_4_only_vertex2_test_set.json")
        if (len(train), len(test)) != (6400, 1600):
            raise RuntimeError(f"Unexpected MetroMap row counts: {len(train)}/{len(test)}")
        print("[shape] metromap train/test=6400/1600")

    if domain in ("all", "travelmap"):
        path = root / "travelmap" / "data" / "all" / "travelmap_shortest_path_query_map_and_tab_with_constraint_1_2_3_4_only_vertex2.json"
        rows = load_json_rows(path)
        train_count = sum(row.get("set_category") == "training_set" for row in rows)
        test_count = sum(row.get("set_category") == "test_set" for row in rows)
        if (train_count, test_count) != (6720, 1680):
            raise RuntimeError(f"Unexpected TravelMap row counts: {train_count}/{test_count}")
        print("[shape] travelmap train/test=6720/1680")


def verify_locked_splits(domain: str) -> None:
    domains = ("metromap", "travelmap") if domain == "all" else (domain,)
    expected_sizes = {"train": 1600, "validation": 100, "test": 400}
    source_split = {"train": "train", "validation": "train", "test": "test"}
    bounds = {
        "metromap": {"train": 6400, "test": 1600},
        "travelmap": {"train": 6720, "test": 1680},
    }
    for current_domain in domains:
        split_sets: dict[str, set[str]] = {}
        split_root = ROOT / "data" / "reference" / current_domain / "splits"
        for split, expected_size in expected_sizes.items():
            ids = json.loads((split_root / f"{split}_sample_ids.json").read_text(encoding="utf-8"))
            if len(ids) != expected_size or len(set(ids)) != expected_size:
                raise RuntimeError(f"Invalid {current_domain}/{split} ID count or duplicates")
            prefix = f"{current_domain}:{source_split[split]}:"
            for sample_id in ids:
                if not sample_id.startswith(prefix):
                    raise RuntimeError(f"Invalid sample ID prefix: {sample_id}")
                index = int(sample_id.rsplit(":", 1)[1])
                if not 0 <= index < bounds[current_domain][source_split[split]]:
                    raise RuntimeError(f"Sample ID out of range: {sample_id}")
            split_sets[split] = set(ids)
        if split_sets["train"] & split_sets["validation"]:
            raise RuntimeError(f"{current_domain} train and validation IDs overlap")
        if split_sets["test"] & (split_sets["train"] | split_sets["validation"]):
            raise RuntimeError(f"{current_domain} test IDs overlap with train IDs")
        print(f"[splits] {current_domain} locked IDs=1600/100/400")


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe archive member in {archive.name}: {member.filename}")
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"[download] reuse {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"[download] {url}")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_public_test(destination: Path, domain: str, dry_run: bool) -> None:
    selected = [item for item in PUBLIC_ARCHIVES if domain == "all" or item.domain == domain]
    if dry_run:
        for item in selected:
            print(f"{PUBLIC_BASE_URL}/{item.name}?download=true")
        print("Dry run only; no files were downloaded.")
        return

    archive_dir = destination / "archives"
    extract_dir = destination / "extracted"
    for item in selected:
        url = f"{PUBLIC_BASE_URL}/{item.name}?download=true"
        archive = archive_dir / item.name
        download_file(url, archive)
        safe_extract(archive, extract_dir)
        print(f"[extract] {item.name}")

    manifest = {
        "source": "https://huggingface.co/datasets/szq-nju/MapTab",
        "revision": PUBLIC_REVISION,
        "scope": "test_only",
        "domain": domain,
        "warning": "This current public release is separate from the historical snapshot and locked Test400 used by this repository.",
        "archives": [{"name": item.name} for item in selected],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Official test-only release ready: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the historical MapTab snapshot or download the current official test-only release."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source", type=Path, help="Legally obtained historical MapTab snapshot or parent directory")
    mode.add_argument("--download-public-test", action="store_true", help="Download the current official test-only release separately")
    parser.add_argument("--destination", type=Path, help="Output root; defaults to ../maptab_data for historical data")
    parser.add_argument("--domain", choices=("all", "metromap", "travelmap"), default="all")
    parser.add_argument("--check-only", action="store_true", help="Verify source or destination without copying")
    parser.add_argument("--dry-run", action="store_true", help="Print official download URLs without downloading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run and not args.download_public_test:
        raise SystemExit("--dry-run is only valid with --download-public-test")

    if args.download_public_test:
        destination = (args.destination or ROOT.parent / "maptab_public_test").expanduser().resolve()
        download_public_test(destination, args.domain, args.dry_run)
        return

    if args.source:
        source = resolve_source(args.source)
        verify_dataset_shape(source, args.domain)
        verify_locked_splits(args.domain)
        if args.check_only:
            print(f"Historical source is compatible: {source}")
            return
        destination = (args.destination or DEFAULT_MAPTAB_ROOT).expanduser().resolve()
        if source == destination:
            raise SystemExit("Source and destination are identical; use --check-only instead.")
        copy_domains(source, destination, args.domain)
    else:
        if not args.check_only:
            raise SystemExit("Choose --source PATH, --download-public-test, or --check-only.")
        destination = (args.destination or DEFAULT_MAPTAB_ROOT).expanduser().resolve()

    verify_dataset_shape(destination, args.domain)
    verify_locked_splits(args.domain)
    print(f"Historical MapTab data ready: {destination}")


if __name__ == "__main__":
    main()
