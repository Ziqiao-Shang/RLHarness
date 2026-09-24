# Data Preparation

This repository does not redistribute raw MapTab images, tables, or question JSON files. It stores only the data protocol required for reproduction:

- `sources.json`: the official repository, official Hugging Face page, and pinned revisions.
- `reference/<domain>/splits/`: fixed Train1600, Val100, and Test400 sample IDs and split manifests for MetroMap and TravelMap.

Runtime raw data is stored in the sibling directory `../maptab_data/` by default. Set `MAPTAB_ROOT` to use another location.

## Exact Historical Reproduction

The current official release contains only test assets and cannot reconstruct the SFT/GRPO training sets used here. Exact reproduction requires a legally obtained historical MapTab snapshot:

```bash
cd RLHarness
bash scripts/prepare_data.sh --source /path/to/MapTab
```

`--source` may point to `MapTab/` itself or to a parent containing `MapTab/` or `data/raw/MapTab/`. The command:

1. Locates the MetroMap and TravelMap directories.
2. Copies the selected domains into `../maptab_data/`.
3. Validates row counts and fixed split IDs.
4. Stops if a target domain already exists instead of silently overwriting it.

Validation only:

```bash
bash scripts/prepare_data.sh --check-only
```

Custom destination or one domain only:

```bash
bash scripts/prepare_data.sh \
  --source /path/to/MapTab \
  --destination /data/maptab_data \
  --domain travelmap

export MAPTAB_ROOT=/data/maptab_data
```

## Download the Current Public Test Set

The current [MapTab Hugging Face release](https://huggingface.co/datasets/szq-nju/MapTab) exposes only MetroMap and TravelMap test assets. Download them to a separate directory with:

```bash
bash scripts/prepare_data.sh --download-public-test
```

The default destination is `../maptab_public_test/`, which stores the original archives, extracted files, and a download manifest. To print download URLs without downloading:

```bash
bash scripts/prepare_data.sh --download-public-test --dry-run
```

This public release is not byte-identical to the historical snapshot. It is therefore not written to `MAPTAB_ROOT` and cannot replace the locked historical Test400.

## Runtime Data

Training commands create the Git-ignored `data/generated/` tree:

```text
generated/
  splits/       fixed train/validation/test materialization
  sft_teacher/  API responses and audit results
  sft/          ShareGPT SFT data
  rl_round1/    first-round GRPO JSONL
  rl_round2/    second-round GRPO JSONL
  travelmap/    equivalent TravelMap runtime tree
```

These files can be rebuilt from the external raw data, fixed IDs, code, and prompts, so they are not release artifacts.

The training workflow never selects new examples from the full raw corpus. The raw snapshot is used only to locate records by reference ID. SkillOpt, the teacher API, SFT, GRPO, skill evolution, and final evaluation are strictly limited to the fixed Train1600, Val100, and Test400. Low-level resampling requires the explicit development-only `--allow-resample` flag and is never used by a public script.

## Required Data Scope

The historical snapshot must include:

- MetroMap only-vertex2: 6,400 training rows and 1,600 test rows.
- TravelMap only-vertex2: 6,720 training rows and 1,680 test rows.
- Map images and vertex tables referenced by those rows.

The historical MetroMap Train/Val split is sample-disjoint but contains overlapping maps, so it is not described as map-disjoint. TravelMap split constraints are recorded in its manifest.
