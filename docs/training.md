# RLHarness Training Guide

This document describes the public training workflow, its inputs and outputs, and resume behavior.

Each stage can be invoked directly through `scripts/*.sh` or, after `python -m pip install -e .`, through `rlharness <stage> --domain <domain>`. The unified CLI only dispatches to the corresponding script; it does not change stage arguments or behavior.

If an original skill prompt already exists, start at Section 3. To recreate the initial skill prompt and optimize it with SkillOpt, follow Sections 2.1 and 2.2. Offline simulation tests validate code paths, not real candidate quality.

## 1. Workflow

```text
Raw data and original 15-skill prompt for the selected domain
  -> fixed Train1600 / Val100 / Test400
  -> optional Train480 and 12-step SkillOpt, selecting the best prompt on Val100
  -> API generation and audit of SFT teacher trajectories
  -> Qwen3.5-9B SFT
  -> first GRPO round with the original 15 skills, steps 0-400
  -> 128 rollout traces and three evolved skill candidates
  -> synchronized How-to-use and few-shot rewrite for every candidate
  -> GPT-5.6-sol selection from Val100 metrics and skill usage
  -> second GRPO round with the selected prompt, steps 400-600
  -> fixed Test400 evaluation
```

`04_evolve_skills.sh` reruns skill candidate research. To reproduce the committed selection without rerunning evolution, use `prompts/student/<domain>/final.txt`. All seven stage commands accept `--domain metromap|travelmap`; the default is MetroMap.

## 2. Environment Setup

```bash
cd RLHarness

# First run: import a legally obtained MapTab snapshot.
bash scripts/prepare_data.sh --source /path/to/MapTab

export QWEN35_MODEL=/path/to/Qwen3.5-9B
export OPENLUX_API_KEY=YOUR_KEY

# Set these only when the environments are not on PATH.
export PYTHON_BIN=/path/to/llamafactory-env/bin/python
export LLAMAFACTORY_CLI=/path/to/llamafactory-env/bin/llamafactory-cli
export VERL_PYTHON=/path/to/verl-env/bin/python
export VLLM_PYTHON=/path/to/verl-env/bin/python
export SKILL_EVOLUTION_PYTHON=/path/to/skill-evolution-env/bin/python

bash scripts/verify_release.sh
```

The default data location is the sibling directory `../maptab_data/`. Set `MAPTAB_ROOT=/path/to/maptab_data` to use another location. The current official MapTab release contains only test assets and cannot replace the complete historical training snapshot required by this workflow. See [`data/README.md`](../data/README.md).

The release includes pinned LLaMA-Factory and VERL source trees, but not raw MapTab assets, Python environments, or Qwen3.5-9B weights.

The environment selected by `SKILL_EVOLUTION_PYTHON` must import `openai` and `Pillow`. If stage 04 must regenerate local rollouts, it must also include the repository's evaluation dependencies: `torch`, `transformers`, and `vllm`.

### 2.1 Optimize the Initial Prompt with SkillOpt

`scripts/00_initialize_skills.sh` chains seed creation from the fixed task frame with SkillOpt optimization. This stage does not generate SFT teacher trajectories. The post-RL 128-trace reconstruction remains a separate stage.

```text
Fixed domain frame without skill, How-to, or few-shot bodies
  -> GPT-5.6 creates 15 skills: five image, five table, and five fusion skills
  -> GPT-5.6 creates How to Use instructions
  -> GPT-5.6 solves two fixed training examples
  -> the program verifies routes, numbers, and skill references
  -> assemble and persist the complete seed prompt
  -> evaluate the baseline on the fixed Val100
  -> run 12 steps, each using one preassigned batch of 40 training questions
     -> Gemini-3.5 student solves image + table + prompt without GT
     -> programmatic scoring separates successful and failed traces
     -> GPT-5.6 reflects with minibatch size 8
     -> merge suggestions with merge batch size 8 and rank them
     -> apply the patch and validate task rules and interface constraints
     -> evaluate the candidate on the same Val100
     -> accept only a strict accuracy increase; ties and decreases retain the current prompt
  -> export the best-on-Val complete prompt and skill JSON
  -> regenerate SFT teacher trajectories with that prompt
```

The default student route is `gemini-3.5-flash`; the default teacher is `gpt-5.6-sol`. Override them with `--student-model` and `--teacher-model` if the provider exposes different route names. Offline tests do not prove that hosted routes are available. Default student and teacher-analysis concurrency are both 8 and are controlled by `--workers` and `--analyst-workers`.

A complete run executes 12 steps. Use `--max-train-steps 1` only for a path-level smoke test. This truncates the current run without changing the fixed Train480 batch order.

Seed creation starts from `prompts/student/<domain>/frame.txt` and does not read the skill or few-shot body from an existing `original.txt`. It uses these templates in order:

```text
prompts/initialization/<domain>/create_skills.txt
prompts/initialization/<domain>/update_usage.txt
prompts/initialization/<domain>/solve_task.txt
```

The default demonstrations are the first two samples in the locked Train480 order. This choice is deterministic and does not claim that the pair is automatically complementary. Use `--demo-ids ID1 ID2` to select two other training samples. Demonstration IDs may not overlap Val100 or Test400. Map overlap is recorded but does not block execution or alter the fixed splits.

Seed calls default to 8,192 output tokens and up to three validation-repair attempts. Override them with `--seed-max-tokens` and `--seed-repair-attempts`. A seed that fails validation stops before SkillOpt. Validation recomputes the GT route and table values and checks output structure and references, but it cannot prove every visual statement or natural-language explanation.

If a complete seed already exists, pass `--initial-prompt /path/to/seed_full_prompt.txt` to skip seed creation and start optimization. It cannot be combined with `--demo-ids`. The command never overwrites committed `original.txt` or `final.txt` files.

Implementation entry points:

- Configuration: `configs/skillopt.yaml`.
- Scheduling and best-result export: `src/rlharness/initialization/skillopt_run.py`.
- Fixed batches, multimodal execution, and scoring: `src/rlharness/initialization/skillopt_adapter.py`.
- Installable optimization engine and teacher prompts: `src/skillopt/`.
- Provenance and local patches: `third_party/SkillOpt/README.md`.

MetroMap uses domain-specific map-analysis prompts. TravelMap uses generic analysis prompts and supplies its own full task context. Teacher reflection receives text traces; the student always receives the image. Candidates preserve skill IDs, authoritative scoring rules, How-to constraints, and the output protocol while allowing edits to skill descriptions, reasoning guidance, and examples. Unlike post-RL evolution, this stage does not freely split, prune, or reconstruct the entire skill bank.

Prepare Train480 as described in Section 2.2, then run an offline check:

```bash
# API/CPU dependencies are sufficient; no local model or GPU is needed.
python -m pip install openai PyYAML numpy json-repair Pillow

DOMAIN=metromap  # or travelmap
bash scripts/00_initialize_skills.sh \
  --domain "$DOMAIN" \
  --split-dir "data/generated/$DOMAIN/skillopt_train480" \
  --output-dir "artifacts/$DOMAIN/skillopt" \
  --dry-run
```

`--dry-run` checks the split, coverage of all 12 batches, prompt placeholders, images, and tables. It calls no API and writes no training output. For a real run, set `OPENLUX_API_KEY` and remove `--dry-run`. Use `OPENLUX_BASE_URL` for a compatible endpoint. Secrets are never accepted through configuration files or command arguments and are not written to results.

Use `INITIALIZATION_PYTHON` to choose the Python interpreter for this stage. Repeating an identical command resumes safely: seed artifacts are restored from `seed/`, SkillOpt resumes from `optimization/`, and completed student responses are cached by their inputs. API failures are not scored as wrong answers, and missing analysis responses are not treated as legitimate no-op edits. Use a new output directory if the frame, examples, prompt, model, data, or code changes. A step with no recommendation or no effective edit may skip candidate validation. Test400 is never accessed.

```text
artifacts/<domain>/skillopt/
  input_manifest.json                    input settings, prompts, asset paths, and split
  seed/initialization/                   API records for skill, How-to, and example creation
  seed/versions/v0000/                   validated complete initial version
  seed/state.json                        seed creation state
  optimization/selection_eval_baseline/ Val100 baseline
  optimization/steps/                    per-step rollout, reflection, merge, rank, candidate, gate
  optimization/skills/                   retained prompt after each step
  optimization/runtime_state.json        SkillOpt resume state
  optimization/history.json              per-step decisions
  optimization/summary.json              optimization summary
  optimization/best_skill.md             best complete prompt exported by the engine
  best/full_prompt.txt                   exact prompt passed to SFT and RL
  best/skills.json                       skills extracted from the same prompt version
```

Hard accuracy uses MapTab `score_route` station normalization, fuzzy matching, and format checks; it is not byte equality. Soft scores are recorded but do not control the hard gate. Because Val100 is repeatedly used for selection, best-on-Val is not an independent test result.

Continue into SFT while retaining the same Train1600, Val100, and Test400:

```bash
DOMAIN=metromap
export ORIGINAL_SKILLS_JSON="$PWD/artifacts/$DOMAIN/skillopt/best/skills.json"
export STUDENT_PROMPT="$PWD/artifacts/$DOMAIN/skillopt/best/full_prompt.txt"
export PROMPT_TEMPLATE="$STUDENT_PROMPT"

# Use isolated generated-data and training directories for each experiment.
export GENERATED_DATA_DIR="$PWD/data/generated/$DOMAIN/skillopt_sft"
export SPLITS_DIR="$GENERATED_DATA_DIR/splits"
export SFT_DATA_DIR="$GENERATED_DATA_DIR/sft"

bash scripts/01_generate_sft_data.sh --domain "$DOMAIN"
bash scripts/02_train_sft.sh --domain "$DOMAIN"
```

Public scripts always materialize the locked Train1600/Val100/Test400 from `data/reference/<domain>/splits`; this source cannot be replaced by an environment variable. Stage 01 still requires the teacher API and base-model tokenizer. Stages 02 and 03 require GPU environments. This stage uses SkillOpt only to select a prompt; SkillOpt exploration traces do not become SFT labels.

API-free regression test:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m unittest discover -s tests -p test_skillopt.py -v
```

Mocked model responses test seed handoff for both domains, caching, the real trainer's 12-step schedule, accept/reject gates, interruption and resume, and secret redaction. They do not evaluate real candidate quality.

Seed creation is implemented by `initialization.seed`. Subsequent reflection, candidate edits, and validation gates are implemented by the bundled SkillOpt engine. No separate window-based initialization path remains in the repository.

### 2.2 Prepare Train480 for SkillOpt

From the existing locked Train1600, select 480 distinct questions in proportion to map and question difficulty. Random seed 42 produces 12 pre-shuffled batches of 40. Val100 and Test400 are reused unchanged; sampling does not depend on model correctness.

```bash
DOMAIN=metromap  # or travelmap
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m rlharness.data_process.skillopt_split \
  --domain "$DOMAIN" \
  --split-dir "data/reference/$DOMAIN/splits" \
  --output-dir "data/generated/$DOMAIN/skillopt_train480"
```

`--split-dir` must point to the repository's locked `*_sample_ids.json` files. Fixed splits cannot be replaced by resampling.

Outputs include the optimization `train_sample_ids.json`, unchanged validation/test ID files, `batches.json`, and a `manifest.json` containing source paths, stratum counts, and overlap statistics. The command rejects sample-ID leakage and refuses to overwrite a different existing output. Map overlap is reported without modifying the locked validation set.

MetroMap Train480 is drawn from the Train1600 later used by RL. It is not fully map-disjoint from Val100: the two sets share 58 maps but no question IDs. Train480 and Test400 share neither question IDs nor maps. MetroMap validation therefore guarantees only sample-level isolation.

This command prepares data without calling an API. Stage 00 then follows `batches.json` exactly, using each 40-question batch once. The full Train1600 remains the SFT/RL set and is not replaced by Train480.

## 3. Fixed Splits and SFT Teacher Trajectories

The public workflow always reuses the committed Train1600/Val100/Test400. It has no public resampling switch and does not accept replacement IDs.

```bash
bash scripts/01_generate_sft_data.sh --domain metromap
# or: bash scripts/01_generate_sft_data.sh --domain travelmap
```

The command:

1. Materializes Train1600, Val100, and Test400 from `data/reference/<domain>/splits`.
2. Validates counts, ID uniqueness, set disjointness, and source identity.
3. Gives the teacher each Train1600 map, vertex table, and original 15-skill prompt.
4. Does not include the GT route in teacher input; GT is used only for post-generation acceptance.
5. Applies domain-specific numerical audits and converts accepted traces to ShareGPT SFT data.
6. Reserves Val100 for model or prompt selection and Test400 for final evaluation.

Domain teacher prompts:

```text
prompts/data_process/teacher_metromap.txt    MetroMap
prompts/data_process/teacher_travelmap.txt   TravelMap
```

Teacher generation supports resume after failures. The first run writes `labels.manifest.json` in the teacher output directory, recording the teacher model, student prompt, skill bank, teacher prompt, fixed sample list, and generation settings. `--resume` appends only when all recorded content is identical. A changed model, prompt, skill bank, sample set, or critical generation setting requires a new `SFT_TEACHER_DIR`.

The default executes at most three generation rounds and then numerically audits accepted traces. SFT construction requires at least 1,395 audited MetroMap labels or 1,233 TravelMap labels. Override these thresholds with `MIN_SFT_LABELS`.

The default teacher is `gpt-5.6-sol`, with concurrency 64 and up to three rounds. Overrides:

```bash
TEACHER_MODEL=your-model \
TEACHER_WORKERS=32 \
TEACHER_ROUNDS=5 \
bash scripts/01_generate_sft_data.sh --domain travelmap
```

Main outputs:

```text
data/generated/splits/                 MetroMap fixed splits and sample IDs
data/generated/sft_teacher/            MetroMap teacher responses and audit results
data/generated/sft_teacher/labels.manifest.json
data/generated/sft/train.json          MetroMap SFT data
data/generated/travelmap/splits/       TravelMap fixed splits and sample IDs
data/generated/travelmap/sft_teacher/  TravelMap teacher responses and audit results
data/generated/travelmap/sft/train.json
```

An identical configuration can resume. A teacher output directory cannot switch models or prompts. Success means that the selected domain's `data/generated[/travelmap]/sft/train.json` exists and is nonempty.

To rematerialize fixed splits only:

```bash
PYTHONPATH=src python -m rlharness.data_process.data_split \
  --domain travelmap \
  --output-dir data/generated/travelmap/splits \
  --fixed-ids-dir data/reference/travelmap/splits
```

## 4. SFT

```bash
bash scripts/02_train_sft.sh --domain metromap
# or: bash scripts/02_train_sft.sh --domain travelmap
```

Settings are in `configs/sft_train.yaml` and `configs/sft_merge.yaml`:

- 2 epochs.
- LoRA rank 16 and alpha 32.
- Learning rate `5e-6`.
- Frozen vision tower and multimodal projector; language-side LoRA only.
- Native thinking disabled.
- Maximum length 24,576.

The launcher writes an experiment-local `llamafactory/dataset_info.json` under the active SFT data directory and passes it through `dataset_dir`. It never modifies `third_party/LlamaFactory/data/dataset_info.json`, so domains and experiments can use separate `SFT_DATA_DIR` and `SFT_REGISTRY_DIR` values without overwriting vendored source.

Outputs:

```text
artifacts/sft/                       SFT LoRA checkpoints
artifacts/sft_merged/                merged SFT model
artifacts/selected_sft_checkpoint.txt
```

TravelMap writes to `artifacts/travelmap/sft/`, `artifacts/travelmap/sft_merged/`, and `artifacts/travelmap/selected_sft_checkpoint.txt`.

Success means `artifacts/sft_merged/config.json` exists for MetroMap or the corresponding TravelMap file exists.

## 5. First GRPO Round

```bash
bash scripts/03_train_rl_round1.sh --domain metromap
```

The defaults are the selected domain's merged SFT model, fixed Train1600/Val100, and `prompts/student/<domain>/original.txt`. Before training, the launcher copies the exact prompt to `selected_prompt.txt` under the round-one output and writes `prompt_manifest.json`. RL data is always constructed from this snapshot. Reusing an output directory with changed prompt content is rejected; set a new `RL_OUTPUT_DIR` instead.

Both domains share `configs/rl_round1.yaml`. The launcher injects domain-specific paths, prompts, and experiment names from `--domain`.

TravelMap:

```bash
bash scripts/03_train_rl_round1.sh --domain travelmap
```

Default TravelMap outputs:

```text
data/generated/travelmap/splits/
data/generated/travelmap/rl_round1/
artifacts/travelmap/grpo_round1/
```

`--domain` is passed to both split materialization and RL data construction. Generated JSONL, manifests, and `prompt_transform_version` record the real domain. Replacing only the prompt while leaving rows labeled as MetroMap is not supported.

Main settings:

- Prompt batch size 8 and 8 responses per question.
- 400 training steps over 2 epochs.
- Language-side LoRA rank 16 with frozen vision components.
- Validation every 10 steps and checkpointing every 40 steps.
- Route reward, DAPO group filtering, and Hybrid-DGPO.

Outputs:

```text
data/generated/rl_round1/
artifacts/grpo_round1/checkpoints/
artifacts/grpo_round1/rollouts/
artifacts/grpo_round1/validation/
artifacts/grpo_round1/selected_prompt.txt
artifacts/grpo_round1/prompt_manifest.json
```

TravelMap adds the `travelmap/` layer, for example `artifacts/travelmap/grpo_round1/`.

Success markers:

```text
artifacts/grpo_round1/checkpoints/global_step_400/data.pt
artifacts/grpo_round1/checkpoints/global_step_400/adapter/adapter_config.json
```

Restarting with the same prompt and output directory lets VERL resume automatically. A different prompt is rejected before RL data generation. See [Reward Design](reward.md) for the reward definition.

## 6. Skill Evolution

```bash
export OPENLUX_API_KEY=YOUR_KEY
bash scripts/04_evolve_skills.sh --domain metromap
```

The workflow is implemented in `src/rlharness/evolution/skill_pipeline.py`.

### 6.1 Rollouts and Pruning

The step-400 checkpoint generates one trajectory for every fixed Train1600 sample with the current complete student prompt. Usage is the union of skill IDs declared under `[Relevant Strategy Selection]` and `[Using Skill N]` markers. Selection counts, marker counts, and union counts are all retained.

Skills with union usage below 0.5% become pruning candidates. Adjust the threshold with `--min-usage-rate` or protect skills with `--force-keep-skills`. Pruning never overwrites the committed prompt.

### 6.2 The 128-Trace Evidence Set

The program samples exactly 128 traces from the same rollout: 64 successful and 64 failed, stratified by task difficulty. Eight mixed batches of 16 are analyzed by `gpt-5.6-sol`. Every reusable conclusion must cite at least two distinct trajectory IDs.

Domain-specific analysis prompts:

```text
prompts/evolution/metromap/analyze.txt
prompts/evolution/travelmap/analyze.txt
```

### 6.3 Three Candidates

The domain's `generate.txt` produces three design directions:

- `candidate_a_conservative`: conservative rewrite.
- `candidate_b_stage_based`: organization by normal reasoning stage.
- `candidate_c_compact`: merge overlapping responsibilities.

Each skill uses `Title / Trigger / Action / Guard / Check`. Skills describe normal forward reasoning and observable high-risk states, rather than recovery rules that require the model to know it has already made an error.

No fixed candidate skill count is imposed. Every candidate must pass two deterministic audits:

- `operation_log`: every source skill is processed exactly once by `KEEP / REWRITE / MERGE / DELETE`; `INSERT` cannot claim a source skill.
- `coverage_audit`: every source responsibility is audited exactly once, its disposition agrees with `operation_log`, result titles exist, and deletions cite evidence.

### 6.4 Synchronize the Complete Prompt

Every candidate independently calls the domain's `rewrite.txt` to update:

- How to Use the Skills.
- `[Relevant Strategy Selection]`.
- `[Using Skill N]` markers in each few-shot example.
- The local reasoning after each marker so that it actually executes the new skill.

The Required Procedure, task facts, numerical values, and final answers must remain unchanged. The program checks each example's selection list, markers, marker audit, natural coverage status, example count, numerical values, and final `<response>`. A skill may be marked `UNILLUSTRATED`; unrelated examples are never padded solely to force coverage. Candidates that fail deterministic validation do not enter Val100 evaluation.

### 6.5 Validation and Automatic Selection

All three complete prompts are evaluated by `qwen3.5-plus` on the same fixed Val100. `gpt-5.6-sol` then reads one normalized `candidate_review.json` and returns a candidate plus a concise reason using:

- Exact, partial, and format accuracy and truncation count.
- Usage and conditional accuracy for each skill.
- Whether any skill is never invoked.
- Whether the candidate introduces an overly broad default behavior.

Test400 is not available to candidate selection.

Main outputs:

```text
artifacts/skill_evolution_source/train_predictions.jsonl
artifacts/skill_evolution/pruning/
artifacts/skill_evolution/evidence128/
artifacts/skill_evolution/candidates/<candidate>/full_prompt.txt
artifacts/skill_evolution/validation_qwen35plus/candidate_review.json
```

Automatic selection is the final step of `04_evolve_skills.sh`. Candidate requests, raw candidate results, few-shot rewrite requests, and the exact full prompt evaluated for each candidate are stored as run snapshots. If selector invocation is interrupted, an identical rerun reuses validated candidates and Val100 results. If a candidate prompt differs from its saved Val100 snapshot by even one byte, old predictions are rejected and a new `SKILL_EVOLUTION_RUN_DIR` is required.

Valid names are `candidate_a_conservative`, `candidate_b_stage_based`, and `candidate_c_compact`. The selector sees only Val100 summaries, never Test400. The program validates the name, few-shot completeness, evaluation domain, and evaluated prompt before writing:

```text
artifacts/skill_evolution/selected/full_prompt.txt
artifacts/skill_evolution/selected/planning_skills.txt
artifacts/skill_evolution/selected/candidate.json
artifacts/skill_evolution/selected/selection.json
```

`selection.json` records the selector model and reason, candidate name, Val100 metrics, skill usage, and source paths for the candidate, report, and complete prompt. An existing selection is not silently rerun; use `--replace` for an intentional reselection. TravelMap uses `artifacts/travelmap/skill_evolution/selected/`.

The committed selections are also stored in `prompts/student/metromap/final.txt` and `prompts/student/travelmap/final.txt` for reproduction without rerunning evolution.

The fixed workflow is:

```text
Current complete prompt and rollout
  -> union statistics over selection and markers
  -> usage-based pruning
  -> stratified sample of 128 traces
  -> eight teacher-analysis batches
  -> whole-bank candidate generation
  -> operation_log and coverage_audit validation
  -> API synchronization of How to Use and few-shot sections
  -> deterministic completeness checks
  -> fixed Val100 evaluation
  -> GPT-5.6-sol selection from accuracy and skill usage
```

The implementation is shared, but the base student prompt and three evolution meta-prompts are domain-specific. `--domain` selects the proper base prompt, meta-prompts, model and adapter defaults, Train1600, Val100, and output directories. Use `--prompt`, `--model`, or `--adapter` only for explicit overrides.

TravelMap:

```bash
bash scripts/04_evolve_skills.sh --domain travelmap
```

This command selects the TravelMap SFT model, first-round step-400 adapter, Train1600/Val100, `prompts/student/travelmap/original.txt`, and `prompts/evolution/travelmap/`. It does not read MetroMap prompts.

## 7. Second GRPO Round

```bash
bash scripts/05_train_rl_round2.sh --domain metromap
```

Prompt resolution order:

1. Explicit `PROMPT_TEMPLATE`.
2. `artifacts[/travelmap]/skill_evolution/selected/full_prompt.txt` with its `selection.json`.
3. Committed `prompts/student/<domain>/final.txt`.

The launcher persists the resolved prompt as `artifacts[/travelmap]/grpo_round2/selected_prompt.txt` and writes `prompt_manifest.json` beside it. Resumed training reads that snapshot. A different requested prompt cannot reuse the output directory.

Training resumes from first-round `global_step_400` and stops at `global_step_600`. Both domains share `configs/rl_round2.yaml`, which inherits `configs/rl_round1.yaml`; the launcher injects domain settings.

TravelMap:

```bash
bash scripts/05_train_rl_round2.sh --domain travelmap
```

It follows the same prompt priority, resumes from `artifacts/travelmap/grpo_round1/checkpoints/global_step_400`, and writes `artifacts/travelmap/grpo_round2/`.

Both round scripts allow overrides through `DOMAIN`, `SPLITS_DIR`, `RL_DATA_DIR`, `RL_OUTPUT_DIR`, `PROMPT_TEMPLATE`, and `SFT_MERGED_MODEL`. Round two also accepts `ROUND1_CHECKPOINT`. Arguments after `--domain` are forwarded as Hydra overrides.

Outputs:

```text
data/generated/rl_round2/
artifacts/grpo_round2/checkpoints/
artifacts/grpo_round2/rollouts/
artifacts/grpo_round2/validation/
artifacts/grpo_round2/selected_prompt.txt
artifacts/grpo_round2/prompt_manifest.json
```

TravelMap writes the equivalent files under `artifacts/travelmap/grpo_round2/`.

Success markers:

```text
artifacts/grpo_round2/checkpoints/global_step_600/data.pt
artifacts/grpo_round2/checkpoints/global_step_600/adapter/adapter_config.json
```

## 8. Fixed Test400

The default evaluates the merged SFT model with the step-600 adapter after validating and loading `selected_prompt.txt` from the round-two directory:

```bash
GPU_IDS="0 1 2 3" bash scripts/06_evaluate_test.sh --domain metromap
```

Explicit model and adapter:

```bash
MODEL_PATH=artifacts/sft_merged \
ADAPTER_PATH=artifacts/grpo_round2/checkpoints/global_step_600/adapter \
OUTPUT_DIR=results/test400_reproduced \
GPU_IDS="0 1 2 3" \
bash scripts/06_evaluate_test.sh
```

TravelMap:

```bash
GPU_IDS="0 1 2 3" \
bash scripts/06_evaluate_test.sh --domain travelmap
```

Outputs:

```text
results/test400/predictions.jsonl
results/test400/predictions.summary.json
results/test400/logs/
```

TravelMap defaults to `results/travelmap/test400/`.

## 9. Failure Recovery

- Teacher API interruption: rerun `01_generate_sft_data.sh`; accepted records are reused.
- GRPO interruption: restart the same command with the same output directory and prompt.
- Skill rollout interruption: retain the output directory and restart; use `--skip-rollout` only when the rollout is complete.
- Changed selector model or selection prompt: use a new `SKILL_EVOLUTION_RUN_DIR` and rerun `04_evolve_skills.sh`; do not overwrite an old selection.
- Evaluation interruption: GPU shards resume from their output files; rerun the evaluation command.
- Never mix different models, prompts, or data splits in one output directory.
