#!/usr/bin/env python3
"""Evaluate Qwen on raw MapTab route-planning data with batched vLLM."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalize_cpu_thread_env(default: int = 8) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        try:
            valid = int(os.environ.get(name, "")) > 0
        except ValueError:
            valid = False
        if not valid:
            os.environ[name] = str(default)


normalize_cpu_thread_env()

from PIL import Image

from rlharness.common.config import ROOT
from rlharness.common.data_load import load_planning
from rlharness.common.prompt_build import build_prompt_bundle
from rlharness.common.response_parse import extract_reasoning, extract_response, extract_think
from rlharness.common.route_score import score_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("all", "metromap", "travelmap"), default="all")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Optional prepared JSON/JSONL rows containing student_prompt, image_path, and gt_route.",
    )
    parser.add_argument(
        "--sample-ids-from",
        type=Path,
        help="Optional JSON/JSONL manifest selecting rows by sample_id in manifest order.",
    )
    parser.add_argument(
        "--skill-bank",
        type=Path,
        help="Optional skill bank injected through the current skill-augmented prompt.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        help="Optional prompt template containing question and weight placeholders.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("MODEL_PATH", ROOT / "artifacts" / "sft_merged")),
        help="Base or merged model path; public scripts select the domain-specific trained output.",
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--visible-reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require generated <reasoning>...</reasoning> before <response>.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "xhigh"),
        default="low",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--image-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--table-max-chars", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--backend",
        choices=("auto", "vllm", "transformers"),
        default=os.environ.get("RLHARNESS_EVAL_BACKEND", "auto"),
        help=(
            "Generation backend. For Qwen3.5, auto uses native vLLM when the "
            "installed version is at least 0.26 and direct Transformers otherwise."
        ),
    )
    parser.add_argument(
        "--model-impl",
        choices=("auto", "vllm", "transformers"),
        default=os.environ.get("RLHARNESS_VLLM_MODEL_IMPL", "auto"),
        help="Implementation override passed to vLLM when --backend=vllm.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-mm-profiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-tower-connector-lora",
        action="store_true",
        help="Allow LoRA tensors from the visual tower or multimodal connector.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start < 0:
        raise SystemExit("--start must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive.")
    if args.table_max_chars <= 0:
        raise SystemExit("--table-max-chars must be positive.")
    if args.max_model_len <= args.max_new_tokens:
        raise SystemExit("--max-model-len must be larger than --max-new-tokens.")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise SystemExit("--gpu-memory-utilization must be between 0 and 1.")


def _read_model_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return config if isinstance(config, dict) else {}


def is_qwen35_model(model_path: Path) -> bool:
    config = _read_model_config(model_path)
    architectures = config.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    return config.get("model_type") == "qwen3_5" or any(
        str(name).startswith("Qwen3_5") for name in architectures
    )


def resolve_eval_backend(model_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if not is_qwen35_model(model_path):
        return "vllm"
    try:
        raw_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return "transformers"
    parts = []
    for value in raw_version.split(".")[:3]:
        digits = "".join(character for character in value if character.isdigit())
        parts.append(int(digits or 0))
    version = tuple((parts + [0, 0, 0])[:3])
    return "vllm" if version >= (0, 26, 0) else "transformers"


def load_rows(domain: str, split: str) -> list[dict[str, Any]]:
    domains = ("metromap", "travelmap") if domain == "all" else (domain,)
    rows: list[dict[str, Any]] = []
    for item in domains:
        rows.extend(load_planning(item, split))
    return rows


def load_input_rows(path: Path, domain: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Input JSON is missing: {path}")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"Input JSON must contain a list: {path}")
        rows = payload
    if domain != "all":
        rows = [row for row in rows if str(row.get("domain")) == domain]
    required = ("sample_id", "student_prompt", "gt_route")
    for index, row in enumerate(rows):
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise SystemExit(f"Missing {missing} in prepared row {index}: {path}")
    return rows


def load_sample_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Sample ID manifest is missing: {path}")
    if path.suffix == ".jsonl":
        payload = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sample_ids = [
            str(item.get("sample_id") if isinstance(item, dict) else item)
            for item in payload
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("sample_ids", []) if isinstance(payload, dict) else payload
        sample_ids = [str(value) for value in values]
    if not sample_ids or any(not value for value in sample_ids):
        raise SystemExit(f"No valid sample_ids found in: {path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit(f"Duplicate sample_ids found in: {path}")
    return sample_ids


def select_sample_ids(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    sample_ids = load_sample_ids(path)
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} sample IDs are absent from the selected split; examples={missing[:5]}"
        )
    return [rows_by_id[sample_id] for sample_id in sample_ids]


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id") or "")
            if not sample_id:
                raise SystemExit(f"Missing sample_id at {path}:{line_number}")
            records[sample_id] = record
    return records


def preprocess_image(
    image_path: str,
    image_max_pixels: int,
    image_min_pixels: int = 3_136,
) -> Image.Image:
    """Match the Stage1/LlamaFactory image resize policy."""
    with Image.open(image_path) as source:
        image = source.copy()
    pixels = image.width * image.height
    if pixels > image_max_pixels:
        factor = math.sqrt(image_max_pixels / pixels)
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    elif pixels < image_min_pixels:
        factor = math.sqrt(image_min_pixels / pixels)
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def build_request(
    processor: Any,
    row: dict[str, Any],
    *,
    thinking: bool,
    reasoning_effort: str,
    image_max_pixels: int,
    table_max_chars: int,
    skill_bank: Any = None,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    if row.get("student_prompt"):
        images = row.get("images") or []
        bundle = {
            "text": str(row["student_prompt"]),
            "image_path": str(row.get("image_path") or (images[0] if images else "")),
        }
    else:
        bundle = build_prompt_bundle(
            row,
            skill_bank=skill_bank,
            include_skills=skill_bank is not None,
            thinking=thinking,
            table_max_chars=table_max_chars,
            prompt_template=prompt_template,
        )
    content: list[dict[str, Any]] = []
    image_path = bundle["image_path"]
    if image_path:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": bundle["text"]})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
    if thinking and prompt.endswith("<think>\n"):
        # Let the model generate its own opening tag, matching LlamaFactory inference.
        prompt = prompt[: -len("<think>\n")]

    request: dict[str, Any] = {"prompt": prompt}
    if image_path:
        request["multi_modal_data"] = {
            "image": preprocess_image(image_path, image_max_pixels)
        }
    return request


def score_prediction(
    prediction: str,
    gt_route: str,
    *,
    thinking: bool,
    visible_reasoning: bool,
) -> dict[str, Any]:
    pred_route = extract_response(prediction) or ""
    thought = extract_think(prediction)
    reasoning = extract_reasoning(prediction)
    scores = score_route(pred_route, gt_route)
    if visible_reasoning:
        format_ok = bool(pred_route) and reasoning is not None and thought is None
    else:
        format_ok = bool(pred_route) and ((thought is not None) if thinking else (thought is None))
    return {
        "think": thought or "",
        "reasoning": reasoning or "",
        "pred_route": pred_route,
        "format_ok": format_ok,
        **scores,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    difficulty_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["domain"])].append(record)
        difficulty = (
            f"{record.get('Map_Difficulty', 'Unknown')}x"
            f"{record.get('Query_Difficulty', 'Unknown')}"
        )
        difficulty_groups[difficulty].append(record)

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        denominator = max(count, 1)
        return {
            "count": count,
            "all_acc": sum(float(row.get("all_acc", 0.0)) for row in rows) / denominator,
            "part_acc": sum(float(row.get("part_acc", 0.0)) for row in rows) / denominator,
            "format_rate": sum(bool(row.get("format_ok")) for row in rows) / denominator,
            "truncated": sum(row.get("finish_reason") == "length" for row in rows),
            "avg_prompt_tokens": sum(int(row.get("prompt_length", 0)) for row in rows) / denominator,
            "max_prompt_tokens": max((int(row.get("prompt_length", 0)) for row in rows), default=0),
            "avg_response_tokens": sum(int(row.get("response_length", 0)) for row in rows) / denominator,
            "max_response_tokens": max((int(row.get("response_length", 0)) for row in rows), default=0),
        }

    return {
        **metrics(records),
        "breakdown": {domain: metrics(rows) for domain, rows in sorted(groups.items())},
        "difficulty_breakdown": {
            difficulty: metrics(rows)
            for difficulty, rows in sorted(difficulty_groups.items())
        },
    }


def write_summary(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    started: float,
) -> Path:
    records = [
        existing[str(row["sample_id"])]
        for row in selected
        if str(row["sample_id"]) in existing and not existing[str(row["sample_id"])].get("error")
    ]
    summary = aggregate(records)
    summary.update(
        {
            "domain": args.domain,
            "split": args.split,
            "input_json": str(args.input_json) if args.input_json else None,
            "sample_ids_from": str(args.sample_ids_from) if args.sample_ids_from else None,
            "skill_bank": str(args.skill_bank) if args.skill_bank else None,
            "prompt_template": str(args.prompt_template) if args.prompt_template else None,
            "model": str(args.model),
            "backend": getattr(args, "resolved_backend", args.backend),
            "model_impl": args.model_impl,
            "adapter": str(args.adapter) if args.adapter else None,
            "thinking": args.thinking,
            "visible_reasoning": args.visible_reasoning,
            "reasoning_effort": args.reasoning_effort if args.thinking else None,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "image_max_pixels": args.image_max_pixels,
            "table_max_chars": args.table_max_chars,
            "selected": len(selected),
            "predictions": str(args.output),
            "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
        }
    )
    path = args.output.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.resolved_backend = resolve_eval_backend(args.model, args.backend)
    if args.input_json is not None and args.sample_ids_from is not None:
        raise SystemExit("Use either --input-json or --sample-ids-from, not both.")
    all_rows = (
        load_input_rows(args.input_json, args.domain)
        if args.input_json
        else load_rows(args.domain, args.split)
    )
    if args.sample_ids_from is not None:
        all_rows = select_sample_ids(all_rows, args.sample_ids_from)
    stop = len(all_rows) if args.limit is None else min(len(all_rows), args.start + args.limit)
    selected = all_rows[args.start:stop]
    selected_ids = {str(row["sample_id"]) for row in selected}
    if not selected:
        raise SystemExit("No planning rows selected.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(args.output) if args.resume else {}
    completed = {
        sample_id: record
        for sample_id, record in existing.items()
        if sample_id in selected_ids and not record.get("error")
    }
    pending = [row for row in selected if str(row["sample_id"]) not in completed]
    mode = "a" if args.resume and args.output.exists() else "w"
    print(
        f"RLHarness rows={len(selected)} complete={len(completed)} pending={len(pending)} "
        f"thinking={args.thinking} backend={args.resolved_backend}",
        flush=True,
    )
    started = time.monotonic()
    if not pending:
        summary = write_summary(args, selected, existing, started)
        print(f"Nothing to resume. Summary: {summary}")
        return

    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    if not args.model.is_dir():
        raise SystemExit(f"Model directory is missing: {args.model}")
    if args.adapter is not None and not (args.adapter / "adapter_config.json").is_file():
        raise SystemExit(f"LoRA adapter is missing or incomplete: {args.adapter}")
    if args.resolved_backend == "transformers" and args.adapter is not None:
        raise SystemExit(
            "The direct Transformers backend accepts merged checkpoints only; "
            "use --backend vllm for LoRA adapters."
        )
    if args.skill_bank is not None and not args.skill_bank.is_file():
        raise SystemExit(f"Skill bank is missing: {args.skill_bank}")
    if args.prompt_template is not None and not args.prompt_template.is_file():
        raise SystemExit(f"Prompt template is missing: {args.prompt_template}")

    skill_bank = (
        json.loads(args.skill_bank.read_text(encoding="utf-8"))
        if args.skill_bank is not None
        else None
    )
    prompt_template = (
        args.prompt_template.read_text(encoding="utf-8").strip()
        if args.prompt_template is not None
        else None
    )

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    effective_batch_size = args.batch_size
    llm = None
    sampling = None
    lora_request = None
    hf_model = None
    if args.resolved_backend == "vllm":
        from vllm import LLM, SamplingParams

        llm_kwargs: dict[str, Any] = {}
        if args.model_impl != "auto":
            llm_kwargs["model_impl"] = args.model_impl
        if args.adapter is not None:
            from vllm.lora.request import LoRARequest

            adapter_config = json.loads(
                (args.adapter / "adapter_config.json").read_text(encoding="utf-8")
            )
            lora_rank = int(adapter_config.get("r", 16))
            llm_kwargs.update(
                enable_lora=True,
                max_loras=1,
                max_lora_rank=lora_rank,
                enable_tower_connector_lora=args.enable_tower_connector_lora,
            )
            lora_request = LoRARequest("rlharness_eval_adapter", 1, str(args.adapter))

        llm = LLM(
            model=str(args.model),
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": 3_136,
                "max_pixels": args.image_max_pixels,
            },
            generation_config="vllm",
            enforce_eager=args.enforce_eager,
            skip_mm_profiling=args.skip_mm_profiling,
            **llm_kwargs,
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            seed=0,
        )
    else:
        from transformers import AutoModelForImageTextToText

        effective_batch_size = 1
        hf_model = AutoModelForImageTextToText.from_pretrained(
            str(args.model),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        hf_model.eval()

    completed_before = len(completed)
    try:
        with args.output.open(mode, encoding="utf-8") as handle:
            for offset in range(0, len(pending), effective_batch_size):
                batch = pending[offset : offset + effective_batch_size]
                batch_started = time.monotonic()
                requests = [
                    build_request(
                        processor,
                        row,
                        thinking=args.thinking,
                        reasoning_effort=args.reasoning_effort,
                        image_max_pixels=args.image_max_pixels,
                        table_max_chars=args.table_max_chars,
                        skill_bank=skill_bank,
                        prompt_template=prompt_template,
                    )
                    for row in batch
                ]
                generated_rows: list[dict[str, Any]] = []
                if args.resolved_backend == "vllm":
                    outputs = llm.generate(
                        requests,
                        sampling_params=sampling,
                        use_tqdm=False,
                        lora_request=lora_request,
                    )
                    for output in outputs:
                        generated = output.outputs[0]
                        generated_rows.append(
                            {
                                "prediction": generated.text,
                                "prompt_length": len(output.prompt_token_ids),
                                "response_length": len(generated.token_ids),
                                "finish_reason": generated.finish_reason,
                            }
                        )
                else:
                    texts = [request["prompt"] for request in requests]
                    images = [
                        request["multi_modal_data"]["image"] for request in requests
                    ]
                    model_inputs = processor(
                        text=texts,
                        images=images,
                        padding=True,
                        return_tensors="pt",
                    )
                    model_inputs = {
                        key: value.to("cuda:0") if hasattr(value, "to") else value
                        for key, value in model_inputs.items()
                    }
                    input_width = int(model_inputs["input_ids"].shape[1])
                    prompt_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
                    with torch.inference_mode():
                        sequences = hf_model.generate(
                            **model_inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                        )
                    token_rows = sequences[:, input_width:]
                    predictions = processor.batch_decode(
                        token_rows,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for prediction, tokens, prompt_length in zip(
                        predictions,
                        token_rows,
                        prompt_lengths,
                        strict=True,
                    ):
                        response_length = int(tokens.shape[0])
                        generated_rows.append(
                            {
                                "prediction": prediction,
                                "prompt_length": int(prompt_length),
                                "response_length": response_length,
                                "finish_reason": (
                                    "length"
                                    if response_length >= args.max_new_tokens
                                    else "stop"
                                ),
                            }
                        )
                    for image in images:
                        image.close()
                seconds_per_sample = (time.monotonic() - batch_started) / len(batch)
                for row, generated in zip(batch, generated_rows, strict=True):
                    prediction = generated["prediction"]
                    scored = score_prediction(
                        prediction,
                        row["gt_route"],
                        thinking=args.thinking,
                        visible_reasoning=args.visible_reasoning,
                    )
                    record = {
                        "sample_id": str(row["sample_id"]),
                        "domain": str(row["domain"]),
                        "split": str(row.get("split") or ("validation" if args.input_json else args.split)),
                        "question": str(row.get("question") or ""),
                        "Map_Difficulty": str(row.get("Map_Difficulty") or "Unknown"),
                        "Query_Difficulty": str(row.get("Query_Difficulty") or "Unknown"),
                        "student_prompt": str(row.get("student_prompt") or ""),
                        "gt_route": str(row["gt_route"]),
                        "prediction": prediction,
                        **scored,
                        "prompt_length": generated["prompt_length"],
                        "response_length": generated["response_length"],
                        "finish_reason": generated["finish_reason"],
                        "seconds": round(seconds_per_sample, 3),
                        "thinking_enabled": args.thinking,
                        "visible_reasoning": args.visible_reasoning,
                        "reasoning_effort": args.reasoning_effort if args.thinking else None,
                        "model": str(args.model),
                        "adapter": str(args.adapter) if args.adapter else None,
                        "backend": args.resolved_backend,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    existing[record["sample_id"]] = record
                handle.flush()
                done = completed_before + offset + len(batch)
                print(f"[{done}/{len(selected)}] batch={len(batch)}", flush=True)
    except KeyboardInterrupt:
        summary = write_summary(args, selected, existing, started)
        print(f"\nInterrupted safely. Partial summary: {summary}")
        raise SystemExit(130)

    summary = write_summary(args, selected, existing, started)
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
