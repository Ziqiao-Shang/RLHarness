#!/usr/bin/env bash
# Verify that the release contains the source artifacts needed for reproduction.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAPTAB_ROOT="${MAPTAB_ROOT:-$ROOT/../maptab_data}"
export MAPTAB_ROOT
errors=0

check() {
  local label="$1" path="$2"
  if [[ -e "$path" ]]; then
    printf 'OK      %-30s %s\n' "$label" "$path"
  else
    printf 'MISSING %-30s %s\n' "$label" "$path"
    errors=$((errors + 1))
  fi
}

optional() {
  local label="$1" path="$2"
  if [[ -e "$path" ]]; then
    printf 'OK      %-30s %s\n' "$label" "$path"
  else
    printf 'OPTIONAL %-30s %s\n' "$label" "$path"
  fi
}

# Scan text artifacts without ever echoing a matched credential value.
PYTHONDONTWRITEBYTECODE=1 python - "$ROOT" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
text_suffixes = {
    ".cfg", ".env", ".ini", ".jinja", ".jinja2", ".json", ".local",
    ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml",
}
token_patterns = (
    ("API-key-like token", re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")),
    ("literal bearer token", re.compile(rb"Bearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I)),
)
assignment = re.compile(
    rb"^[ \t]*(?:export[ \t]+)?(?:OPENLUX_API_KEY|WANDB_API_KEY|API_KEY)"
    rb"[ \t]*=[ \t]*[\"']?([^ \t\r\n\"']*)",
    re.M,
)
placeholder_fragments = (
    b"YOUR", b"REPLACE", b"EXAMPLE", b"PLACEHOLDER", b"CHANGEME",
)
findings: list[tuple[Path, int, str]] = []

for path in root.rglob("*"):
    if not path.is_file() or path.stat().st_size > 5_000_000:
        continue
    if path.suffix.lower() not in text_suffixes and not path.name.startswith(".env"):
        continue
    try:
        data = path.read_bytes()
    except OSError:
        continue
    if b"\0" in data[:4096]:
        continue
    for label, pattern in token_patterns:
        for match in pattern.finditer(data):
            findings.append((path, data.count(b"\n", 0, match.start()) + 1, label))
    if "third_party" not in path.parts or path.name.startswith(".env"):
        for match in assignment.finditer(data):
            value = match.group(1).strip()
            upper = value.upper()
            if (
                not value
                or value.startswith((b"$", b"<", b"{", b"os.environ", b"getenv(", b"str(", b"None"))
                or any(fragment in upper for fragment in placeholder_fragments)
            ):
                continue
            findings.append(
                (path, data.count(b"\n", 0, match.start()) + 1, "literal API-key assignment")
            )

if findings:
    for path, line, label in findings:
        print(f"SECRET  {path.relative_to(root)}:{line}: {label}", file=sys.stderr)
    raise SystemExit("Preflight stopped: remove hardcoded credentials before publishing.")

print("SECRETS no hardcoded API keys or bearer tokens detected")
PY

check "MapTab source metadata" "$ROOT/data/sources.json"
check "Verified environment table" "$ROOT/docs/environment.md"
optional "External MapTab data" "$MAPTAB_ROOT"
check "MetroMap split protocol" "$ROOT/data/reference/metromap/splits/manifest.json"
check "TravelMap split protocol" "$ROOT/data/reference/travelmap/splits/manifest.json"
for data_domain in metromap travelmap; do
  if [[ -d "$MAPTAB_ROOT/$data_domain" ]]; then
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" python -m \
      rlharness.data_process.prepare_data \
      --destination "$MAPTAB_ROOT" --domain "$data_domain" --check-only
  fi
done
check "Installed SkillOpt engine" "$ROOT/src/skillopt/engine/trainer.py"
check "SkillOpt license" "$ROOT/third_party/SkillOpt/LICENSE"
check "SkillOpt config" "$ROOT/configs/skillopt.yaml"
for domain in metromap travelmap; do
  for split in train validation test; do
    check "Locked $domain/$split" "$ROOT/data/reference/$domain/splits/${split}_sample_ids.json"
  done
done

for prompt in \
  student/metromap/original.txt \
  student/metromap/final.txt \
  student/metromap/original_skills.json \
  student/travelmap/original.txt \
  student/travelmap/final.txt \
  student/travelmap/original_skills.json \
  common/task_base.txt \
  common/skill_injection.txt \
  data_process/teacher_metromap.txt \
  data_process/teacher_travelmap.txt \
  evolution/metromap/analyze.txt \
  evolution/metromap/generate.txt \
  evolution/metromap/rewrite.txt \
  evolution/metromap/select.txt \
  evolution/travelmap/analyze.txt \
  evolution/travelmap/generate.txt \
  evolution/travelmap/rewrite.txt \
  evolution/travelmap/select.txt; do
  check "Prompt $prompt" "$ROOT/prompts/$prompt"
done

for config in skillopt.yaml sft_train.yaml sft_merge.yaml rl_round1.yaml rl_round2.yaml; do
  check "Config $config" "$ROOT/configs/$config"
done

for step in \
  prepare_data \
  verify_release \
  00_initialize_skills \
  01_generate_sft_data \
  02_train_sft \
  03_train_rl_round1 \
  04_evolve_skills \
  05_train_rl_round2 \
  06_evaluate_test \
  07_evaluate_pretrained; do
  check "Command $step" "$ROOT/scripts/$step.sh"
done

for module in \
  initialization/__init__.py initialization/state.py initialization/execution.py \
  initialization/seed.py \
  initialization/skillopt_run.py initialization/skillopt_adapter.py \
  cli.py release.py common/api_client.py common/config.py common/data_load.py common/prompt_build.py \
  common/response_parse.py common/route_score.py common/skill_format.py \
  data_process/prepare_data.py data_process/data_split.py data_process/rl_data.py data_process/rl_format.py \
  data_process/sft_audit.py data_process/sft_build.py data_process/sft_teacher.py data_process/skillopt_split.py \
  rl/advantage.py rl/reward.py rl/rl_dataset.py rl/rl_launcher.py \
  eval/eval_api.py eval/eval_local.py eval/eval_merge.py model_release.py \
  evolution/skill_evidence.py \
  evolution/skill_generate.py evolution/skill_pipeline.py \
  evolution/skill_select.py \
  evolution/skill_prune.py evolution/skill_report.py evolution/skill_rewrite.py; do
  check "Python module $module" "$ROOT/src/rlharness/$module"
done

for domain in metromap travelmap; do
  check "Student frame $domain" "$ROOT/prompts/student/$domain/frame.txt"
  for name in create_skills update_usage solve_task; do
    check "Initialization $domain/$name" "$ROOT/prompts/initialization/$domain/$name.txt"
  done
done

for legacy in \
  src/rlharness/initialization/pipeline.py \
  src/rlharness/initialization/handoff.py \
  prompts/initialization/metromap/update_skills.txt \
  prompts/initialization/travelmap/update_skills.txt; do
  if [[ -e "$ROOT/$legacy" ]]; then
    printf 'LEGACY  %-30s %s\n' "Removed flow" "$ROOT/$legacy"
    errors=$((errors + 1))
  fi
done

if grep -q 'third_party/LlamaFactory/data/dataset_info.json' "$ROOT/scripts/02_train_sft.sh"; then
  printf 'LEGACY  %-30s %s\n' "Mutable SFT registry" "$ROOT/scripts/02_train_sft.sh"
  errors=$((errors + 1))
else
  printf 'OK      %-30s %s\n' "Runtime SFT registry" "$ROOT/scripts/02_train_sft.sh"
fi

check "LLaMA-Factory source" "$ROOT/third_party/LlamaFactory/pyproject.toml"
check "VERL source" "$ROOT/third_party/verl-modern/pyproject.toml"
optional "Qwen3.5-9B base model" "${QWEN35_MODEL:-$ROOT/models/Qwen3.5-9B}/config.json"
optional "External MetroMap final model" "${METROMAP_FINAL_MODEL:-${MODEL_PATH:-$ROOT/../metromap_final}}/config.json"
optional "Generated SFT data" "$ROOT/data/generated/sft/train.json"
optional "Generated TravelMap SFT data" "$ROOT/data/generated/travelmap/sft/train.json"
optional "Round-1 checkpoint" "$ROOT/artifacts/grpo_round1/checkpoints/global_step_400/data.pt"
optional "Round-2 checkpoint" "$ROOT/artifacts/grpo_round2/checkpoints/global_step_600/data.pt"
optional "External TravelMap final model" "${TRAVELMAP_FINAL_MODEL:-$ROOT/../travelmap_final}/config.json"
optional "TravelMap Round-1 checkpoint" "$ROOT/artifacts/travelmap/grpo_round1/checkpoints/global_step_400/data.pt"
optional "TravelMap Round-2 checkpoint" "$ROOT/artifacts/travelmap/grpo_round2/checkpoints/global_step_600/data.pt"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" python - "$ROOT" "$MAPTAB_ROOT" <<'PY'
import ast
import json
import sys
from pathlib import Path

from rlharness.common.data_load import load_planning
from rlharness.initialization.state import extract_frame

root = Path(sys.argv[1])
maptab_root = Path(sys.argv[2])
skills = json.loads((root / "prompts/student/metromap/original_skills.json").read_text(encoding="utf-8"))
travel_skills = json.loads((root / "prompts/student/travelmap/original_skills.json").read_text(encoding="utf-8"))
original = (root / "prompts/student/metromap/original.txt").read_text(encoding="utf-8")
final = (root / "prompts/student/metromap/final.txt").read_text(encoding="utf-8")
travel_original = (root / "prompts/student/travelmap/original.txt").read_text(encoding="utf-8")
travel_final = (root / "prompts/student/travelmap/final.txt").read_text(encoding="utf-8")

assert len(skills["skills"]) == 15, len(skills["skills"])
assert len(travel_skills["skills"]) == 15, len(travel_skills["skills"])
assert travel_skills["domain"] == "travelmap"
assert [item["skill_id"] for item in travel_skills["skills"]] == list(range(1, 16))
for item in travel_skills["skills"]:
    for key in ("title", "when_to_apply", "procedure", "check"):
        assert item[key] in travel_original, (item["skill_id"], key)

available_domains = [
    domain for domain in ("metromap", "travelmap")
    if (maptab_root / domain).is_dir()
]
if "metromap" in available_domains:
    assert len(load_planning("metromap", "train")) == 6400
    assert len(load_planning("metromap", "test")) == 1600
    print("COUNTS  metromap=6400/1600")
if "travelmap" in available_domains:
    assert len(load_planning("travelmap", "train")) == 6720
    assert len(load_planning("travelmap", "test")) == 1680
    print("COUNTS  travelmap=6720/1680")
if not available_domains:
    print(f"DATA    external MapTab not present at {maptab_root}; run scripts/prepare_data.sh")

for domain in ("metromap", "travelmap"):
    reference = root / "data/reference" / domain / "splits"
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["train_rows"], manifest["validation_rows"], manifest["test_rows"]) == (1600, 100, 400)
    split_ids = {
        name: json.loads((reference / f"{name}_sample_ids.json").read_text(encoding="utf-8"))
        for name in ("train", "validation", "test")
    }
    assert tuple(map(len, split_ids.values())) == (1600, 100, 400)
    assert not (set(split_ids["train"]) & set(split_ids["validation"]))
    assert not (set(split_ids["train"]) & set(split_ids["test"]))
    assert not (set(split_ids["validation"]) & set(split_ids["test"]))
required_sections = (
    "# Planning Skills",
    "# How to Use the Skills",
    "# Required Procedure",
    "# Few-Shot Demonstrations",
    "# Output Format Constraints",
    "[Relevant Strategy Selection]",
    "<response>",
)
for text in (original, final, travel_original, travel_final):
    assert all(section in text for section in required_sections)

for domain, text in (("metromap", original), ("travelmap", travel_original)):
    frame = (root / "prompts/student" / domain / "frame.txt").read_text(encoding="utf-8")
    assert extract_frame(text).rstrip() == frame.rstrip(), (domain, "immutable initialization frame")

domain_scripts = {
    name: (root / "scripts" / name).read_text(encoding="utf-8")
    for name in (
        "01_generate_sft_data.sh",
        "02_train_sft.sh",
        "03_train_rl_round1.sh",
        "04_evolve_skills.sh",
        "05_train_rl_round2.sh",
        "06_evaluate_test.sh",
    )
}
for name, script in domain_scripts.items():
    assert "--domain" in script and "travelmap" in script, name
assert 'prompts/student/$DOMAIN/original.txt' in domain_scripts["03_train_rl_round1.sh"]
assert 'prompts/student/$DOMAIN/final.txt' in domain_scripts["05_train_rl_round2.sh"]
for name in ("01_generate_sft_data.sh", "03_train_rl_round1.sh", "04_evolve_skills.sh", "05_train_rl_round2.sh", "06_evaluate_test.sh"):
    assert '--domain "$DOMAIN"' in domain_scripts[name], name

required_placeholders = {
    "analyze.txt": ("{skill_bank}", "{trajectories}"),
    "generate.txt": ("{skill_bank}", "{evidence}", "{candidate_name}", "{candidate_style}"),
    "rewrite.txt": (
        "{new_skill_bank}",
        "{operation_log}",
        "{original_how_to_use}",
        "{original_few_shots}",
    ),
    "select.txt": ("{candidate_review}",),
}
for domain in ("metromap", "travelmap"):
    for filename, placeholders in required_placeholders.items():
        text = (root / "prompts/evolution" / domain / filename).read_text(encoding="utf-8")
        assert all(placeholder in text for placeholder in placeholders), (domain, filename)

for package in ("rlharness", "skillopt"):
    for path in (root / "src" / package).rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

print("SKILLS  original skill banks=15+15")
print("SPLITS  MetroMap/TravelMap reference IDs=1600/100/400, pairwise disjoint")
print("SYNTAX  Python package parsed successfully")
PY

for script in "$ROOT"/scripts/*.sh; do
  bash -n "$script"
done
echo "SYNTAX  Shell commands parsed successfully"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" \
  python -m rlharness.release --root "$ROOT"

if ((errors)); then
  echo "Preflight failed with $errors missing file(s)." >&2
  exit 1
fi

echo "Preflight passed. External data/models and generated training artifacts are intentionally optional."
