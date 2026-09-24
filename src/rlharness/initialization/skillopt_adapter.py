"""MapTab bridge to the vendored SkillOpt trainer; no outer-project imports."""

from __future__ import annotations

import importlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from skillopt.datasets.base import BaseDataLoader, BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.model import chat_target_messages

from rlharness.common.api_client import multimodal_messages
from rlharness.common.config import ROOT
from rlharness.common.data_load import load_planning
from rlharness.common.prompt_build import (
    TABLE_INTRO, fill_official_prompt, load_table_text, resolve_image,
)
from rlharness.common.response_parse import parse_reasoning_response, parse_think_response
from rlharness.common.route_score import score_route
from rlharness.data_process.skillopt_split import read_ids
from rlharness.initialization.state import read_json, write_json


class LockedMapLoader(BaseDataLoader):
    def __init__(self, domain: str, split_dir: Path):
        manifest = read_json(split_dir / "manifest.json")
        if manifest["domain"] != domain:
            raise ValueError("SkillOpt split domain mismatch")
        train = read_ids(split_dir / "train_sample_ids.json")
        val = read_ids(split_dir / "validation_sample_ids.json")
        test = read_ids(split_dir / "test_sample_ids.json")
        reference_dir = ROOT / "data" / "reference" / domain / "splits"
        reference_train = read_ids(reference_dir / "train_sample_ids.json")
        reference_val = read_ids(reference_dir / "validation_sample_ids.json")
        reference_test = read_ids(reference_dir / "test_sample_ids.json")
        if not set(train) <= set(reference_train) or val != reference_val or test != reference_test:
            raise ValueError(
                "SkillOpt may only use a Train480 subset of the locked Train1600, "
                "plus the exact Val100/Test400 IDs"
            )
        if len(train) != 480 or len(val) != 100:
            raise ValueError("Expected locked Train480 and Val100")
        if set(train) & set(val) or set(train + val) & set(test):
            raise ValueError("Train/validation/test IDs overlap")
        self.batches = read_json(split_dir / "batches.json")
        if (len(self.batches) != 12 or any(len(b) != 40 for b in self.batches)
                or [sid for b in self.batches for sid in b] != train):
            raise ValueError("Expected an exact, ordered 12 x 40 partition of Train480")
        # Only training-source rows are loaded; test IDs are used for leakage checks.
        rows = {r["sample_id"]: r for r in load_planning(domain, "train")}
        if not set(train + val) <= rows.keys():
            raise ValueError("Unknown or non-training-source optimization/validation ID")
        self.rows = {sid: dict(rows[sid], id=sid) for sid in train + val}
        self.train_items = [self.rows[sid] for sid in train]
        self.val_items = [self.rows[sid] for sid in val]
        self.manifest = manifest

    def get_train_size(self):
        return len(self.train_items)

    def plan_train_epoch(self, *, epoch, steps_per_epoch, accumulation, batch_size, seed, **kwargs):
        if (epoch, steps_per_epoch, accumulation, batch_size) != (1, 12, 1, 40):
            raise ValueError("Locked protocol: one epoch, twelve distinct batches of forty")
        return [BatchSpec("train", "train", seed + i + 1, 40,
                          [self.rows[sid] for sid in batch])
                for i, batch in enumerate(self.batches)]

    def build_train_batch(self, batch_size, seed, **kwargs):
        raise ValueError("Use the locked batches.json plan, not independent resampling")

    def build_eval_batch(self, env_num, split, seed, **kwargs):
        if split not in ("val", "valid_seen") or env_num != len(self.val_items):
            raise ValueError("Only the complete fixed Val100 may be used by SkillOpt")
        return BatchSpec("eval", "val", seed, env_num, self.val_items)


def skill_strings(prompt: str) -> list[str]:
    match = re.search(r"^# Planning Skills\s*\n(.*?)(?=^# )", prompt, re.M | re.S)
    if not match:
        raise ValueError("Missing Planning Skills section")
    items = re.findall(r"^\d+\.\s+(.*?)(?=^\d+\.\s|\Z)", match[1], re.M | re.S)
    if not items:
        raise ValueError("Empty skill bank")
    return [item.strip() for item in items]


class MapSkillOptAdapter(EnvAdapter):
    def __init__(self, domain: str, loader: LockedMapLoader):
        self.domain = domain
        self.loader = loader

    @property
    def _env_name(self):
        return self.domain

    def setup(self, cfg):
        super().setup(cfg)
        self.baseline = Path(cfg["skill_init"]).read_text(encoding="utf-8")
        for key in ("analyst_workers", "failure_only", "minibatch_size", "edit_budget"):
            setattr(self, key, cfg[key])
        self.analyst_response_attempts = int(cfg.get("analyst_response_attempts", 3))
        if self.analyst_response_attempts < 1:
            raise ValueError("analyst_response_attempts must be positive")

    def get_dataloader(self):
        return self.loader

    def build_env_from_batch(self, batch, **kwargs):
        return batch.payload

    def build_train_env(self, batch_size, seed, **kwargs):
        raise ValueError("Independent batch sampling is disabled")

    def build_eval_env(self, env_num, split, seed, **kwargs):
        return self.loader.build_eval_batch(env_num, split, seed).payload

    def get_task_types(self):
        return [self.domain]

    def validate_candidate_skill(self, candidate, **kwargs):
        validator = importlib.import_module(f"skillopt.envs.{self.domain}.prompt_invariants")
        valid, reasons = validator.validate_prompt_candidate(candidate, self.baseline)
        try:
            candidate.format(question="Question", w1=1, w2=0, w3=0, w4=0)
            skill_strings(candidate)
        except (ValueError, KeyError, IndexError) as exc:
            reasons.append(f"Malformed prompt template: {type(exc).__name__}")
        return valid and not reasons, reasons

    def build_reference_text(self, item):
        return f"Ground-truth route (teacher only): {item['gt_route']}"

    def reflect(self, results, skill_content, out_dir, **kwargs):
        # Missing API/JSON responses must not masquerade as a deliberate no-op.
        successes = sum(bool(r["hard"]) for r in results)
        failures = len(results) - successes
        size = self.minibatch_size
        expected = (failures + size - 1) // size
        if not self.failure_only:
            expected += (successes + size - 1) // size
        for attempt in range(1, self.analyst_response_attempts + 1):
            patches = super().reflect(results, skill_content, out_dir, **kwargs)
            if len(patches) == expected:
                return patches
            if attempt < self.analyst_response_attempts:
                print(
                    "      [analyst retry] "
                    f"received {len(patches)}/{expected} valid minibatch responses; "
                    "retrying only uncached minibatches"
                )
        raise RuntimeError(
            "Incomplete analyst responses after "
            f"{self.analyst_response_attempts} attempts; rerun to resume, not skip this step"
        )

    def rollout(self, env_manager, skill_content, out_dir, **kwargs):
        cfg = self._cfg
        root = Path(out_dir)

        def solve(row):
            folder = root / "predictions" / row["id"]
            text = fill_official_prompt(row, prompt_template=skill_content)
            table = load_table_text(rec=row, max_chars=10**9)
            image = resolve_image(row.get("figure"))
            if not image or not table or not row.get("gt_route"):
                raise ValueError(f"Missing image/table/GT for {row['id']}")
            text += f"\n\n{TABLE_INTRO[self.domain]}\n{table}"
            request = {
                "prompt": text,
                "row": row,
                "model": cfg["target_model"],
                "max_tokens": cfg["rollout_max_tokens"],
            }
            cached = folder / "result.json"
            if cached.exists():
                result = read_json(cached)
                if result.get("request") != request:
                    raise ValueError("Cached rollout belongs to a different input/configuration")
                return result
            output, _ = chat_target_messages(
                multimodal_messages(system="", user_text=text, image_path=image),
                max_completion_tokens=cfg["rollout_max_tokens"],
                retries=cfg["rollout_retries"], timeout=cfg["rollout_timeout"], stage="rollout",
            )
            if not str(output).strip():
                raise RuntimeError("Empty student API response; no correctness score recorded")
            parsed = (parse_reasoning_response(output) if "<reasoning>" in skill_content
                      else parse_think_response(output))
            score = score_route(parsed.final or "", row["gt_route"])
            hard = int(parsed.format_ok and score["all_acc"] == 1.0)
            result = {
                "id": row["id"], "request": request, "hard": hard,
                "soft": .70 * score["all_acc"] + .25 * score["part_acc"] + .05 * parsed.format_ok,
                **score, "format_ok": parsed.format_ok, "response": output,
                "ground_truth": row["gt_route"], "task_type": self.domain,
                "task_description": row["question"], "n_turns": 1,
                "fail_reason": "" if hard else ("format" if not parsed.format_ok else "route"),
                "reference_text": self.build_reference_text(row), "image_paths": [image],
            }
            write_json(folder / "conversation.json", [
                {"role": "user", "content": text},
                {"role": "assistant", "content": output},
                {"role": "system", "content": f"GT route: {row['gt_route']}; hard={hard}"},
            ])
            write_json(cached, result)
            return result

        with ThreadPoolExecutor(max_workers=cfg["rollout_workers"]) as pool:
            results = list(pool.map(solve, env_manager))
        write_json(root / "results.json", results)
        return results
