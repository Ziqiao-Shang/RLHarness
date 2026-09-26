# Data Preparation

This repository does not redistribute raw MapTab images, tables, or question JSON files. It stores only the data protocol required for reproduction:

- `sources.json`: the official repository, official Hugging Face page, and pinned revisions.
- `reference/<domain>/splits/`: fixed Train1600, Val100, and Test400 sample IDs and split manifests for MetroMap and TravelMap.

Runtime raw data is stored in the sibling directory `../maptab_data/` by default. Set `MAPTAB_ROOT` to use another location.

## Download by Locked Sample ID

The current [MapTab Hugging Face release](https://huggingface.co/datasets/szq-nju/MapTab) contains the complete MetroMap and TravelMap planning train/test sources. RLHarness pins the official revision, interprets each committed ID as `<domain>:<source-split>:<row-index>`, and downloads only the images and tables referenced by Train1600, Val100, Test400, and reserved prompt examples:

```bash
cd RLHarness
bash scripts/prepare_data.sh --download
```

The command:

1. Downloads the two only-vertex2 source JSON files for each selected domain.
2. Checks the expected 6,400/1,600 MetroMap and 6,720/1,680 TravelMap row counts.
3. Resolves the committed IDs against the original source row order.
4. Downloads the unique referenced images/tables into `../maptab_data/`.
5. Records the repository commit and materialized files under `.rlharness/`.

The download is resumable and reuses the Hugging Face cache. Select one domain or another destination with:

```bash
bash scripts/prepare_data.sh --download \
  --domain travelmap \
  --destination /data/maptab_data

export MAPTAB_ROOT=/data/maptab_data
```

## Lightweight Interface Check

The smoke test downloads only a few source rows and their assets; it does not materialize the full locked subset:

```bash
bash scripts/prepare_data.sh --smoke-test --domain all
# Increase to two train and two test examples per domain:
bash scripts/prepare_data.sh --smoke-test --domain all --smoke-count 2
```

For every sampled ID it verifies source indexing, parses the JSON table, and asks Pillow to verify the downloaded image.

## Import an Existing MapTab Tree

`--source` may point to `MapTab/` itself or to a parent containing `MapTab/` or `data/raw/MapTab/`:

```bash
bash scripts/prepare_data.sh --source /path/to/MapTab
```

The importer copies the selected domain and applies the same row-count, ID, and asset checks.

Validation only:

```bash
bash scripts/prepare_data.sh --check-only
```

To inspect the pinned source paths and counts without downloading:

```bash
bash scripts/prepare_data.sh --download --dry-run
```

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

The source metadata contains:

- MetroMap only-vertex2: 6,400 training rows and 1,600 test rows.
- TravelMap only-vertex2: 6,720 training rows and 1,680 test rows.
- Map images and vertex tables are materialized only for the committed IDs.

The MetroMap Train/Val split is sample-disjoint but contains overlapping maps, so it is not described as map-disjoint. TravelMap split constraints are recorded in its manifest.
