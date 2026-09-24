"""Multimodal calls and explicit, limited deterministic trajectory checks."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from rlharness.common.api_client import (
    chat_text, encode_image_data_url, extract_json_obj, openlux_client,
)
from rlharness.common.prompt_build import build_prompt_bundle, _weights_tuple
from rlharness.data_process.sft_audit import (
    _load_vertex_rows, audit_claimed_numbers, compute_route_metrics,
)
from rlharness.data_process.sft_teacher import EXACT_OUTPUT_RE, REQUIRED_REASONING_SECTIONS
from rlharness.evolution.skill_rewrite import MARKER_RE, selection_ids
from .state import read_json, write_json


def redact(text: str) -> str:
    key = os.environ.get("OPENLUX_API_KEY")
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{20,}", "[REDACTED]", text)


class APIBackend:
    """Cache completed calls; persist error types, never SDK headers or secrets."""

    def __init__(self, timeout: float, max_tokens: int, retries: int = 3,
                 image_max_pixels: int = 1_000_000):
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries
        self.image_max_pixels = image_max_pixels

    def request(self, path: Path, *, model: str, instruction: str,
                payload: dict, images: list[str]) -> Any:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        request = {
            "model": model,
            "instruction": instruction,
            "payload": payload,
            "images": [str(Path(image).resolve()) for image in images],
            "max_tokens": self.max_tokens,
            "image_max_pixels": self.image_max_pixels,
        }
        if path.is_file():
            record = read_json(path)
            if record.get("request") != request:
                raise ValueError(f"Cached request configuration changed: {path.name}")
            return extract_json_obj(record["raw"])
        content: list[dict] = [{"type": "text", "text": serialized}]
        for index, image in enumerate(images, 1):
            from rlharness.eval.eval_api import prepare_image
            cache = path.parent / "images" / f"{index:02d}_{Path(image).stem}_{self.image_max_pixels}"
            url = encode_image_data_url(prepare_image(image, cache, self.image_max_pixels))
            if url is None:
                raise FileNotFoundError("Required image is missing")
            content.extend([
                {"type": "text", "text": f"Attached image {index}"},
                {"type": "image_url", "image_url": {"url": url}},
            ])
        messages = [{"role": "system", "content": instruction},
                    {"role": "user", "content": content}]
        for attempt in range(self.retries):
            try:
                with openlux_client(timeout=self.timeout) as client:
                    raw = chat_text(client, model, messages, temperature=0, max_tokens=self.max_tokens)
                write_json(path, {"request": request, "raw": redact(raw)})
                return extract_json_obj(raw)
            except Exception as exc:
                # Do not serialize the exception message: providers may echo credentials.
                write_json(path.with_suffix(".error.json"), {
                    "error_type": type(exc).__name__, "attempt": attempt + 1,
                })
                if attempt + 1 == self.retries:
                    raise RuntimeError("API request failed; see redacted error record and resume later") from None
                time.sleep(min(2 ** attempt, 8))


def bundle_for(row: dict, prompt: str) -> dict:
    bundle = build_prompt_bundle(row, prompt_template=prompt, thinking=False, table_max_chars=20_000)
    if not bundle["image_path"] or not Path(bundle["image_path"]).is_file():
        raise ValueError(f"Missing map image: {row['sample_id']}")
    if not bundle["table_path"] or not Path(bundle["table_path"]).is_file():
        raise ValueError(f"Missing vertex table: {row['sample_id']}")
    return bundle


def verify(value: Any, row: dict, bundle: dict, skill_count: int) -> dict:
    errors: list[str] = []
    output = value.get("output", "") if isinstance(value, dict) else ""
    if not isinstance(output, str):
        output = ""
    match = EXACT_OUTPUT_RE.fullmatch(output)
    selected: list[int] = []
    used: list[int] = []
    route_correct = False
    numeric = None
    if not match:
        errors.append("Require exactly one reasoning block followed by one response block")
    else:
        reasoning, route = (part.strip() for part in match.groups())
        route_correct = route == str(row["gt_route"]).strip()
        if not route_correct:
            errors.append("Final route does not match the reference answer; recheck the map and calculation")
        if any(reasoning.count(section) != 1 for section in REQUIRED_REASONING_SECTIONS):
            errors.append("Require the five reasoning sections exactly once")
        elif [reasoning.index(s) for s in REQUIRED_REASONING_SECTIONS] != sorted(
            reasoning.index(s) for s in REQUIRED_REASONING_SECTIONS
        ):
            errors.append("Reasoning sections are out of order")
        selected = selection_ids(reasoning)
        used = [int(number) for number in MARKER_RE.findall(reasoning)]
        if not selected or not used or set(selected) != set(used):
            errors.append("Selected and used skill sets must be nonempty and equal")
        if len(selected) != len(set(selected)) or len(used) != len(set(used)):
            errors.append("Duplicate selection or usage marker")
        if any(not 1 <= number <= skill_count for number in selected + used):
            errors.append("Unknown skill number")
        if re.search(r"ground[- ]truth|according to (?:the )?gt\b", reasoning, re.I):
            errors.append("Do not quote reference-answer supervision in the reasoning")
        claims = value.get("metrics")
        try:
            table = _load_vertex_rows(Path(bundle["table_path"]))
            metrics = compute_route_metrics(route, table, list(_weights_tuple(row)), domain=row["domain"])
            expected = dict(T=metrics.t, P=metrics.p, C=metrics.c, R=metrics.r, Score=metrics.score)
            if not isinstance(claims, dict) or not all(
                key in claims and type(claims[key]) in (int, float)
                and math.isfinite(claims[key])
                and abs(claims[key] - number) <= 0.002
                for key, number in expected.items()
            ):
                errors.append("metrics must contain reproducible T/P/C/R/Score for the selected route")
            numeric = audit_claimed_numbers(reasoning, metrics, score_tolerance=0.002,
                                            component_tolerance=0.002)
            if not numeric["passed"]:
                errors.append("Selected-route component values must also appear in reasoning")
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            errors.append("Cannot recompute the selected route from canonical table rows")
    return {"passed": not errors, "errors": errors, "route_correct": route_correct,
            "selected": selected, "used": used, "numeric_audit": numeric,
            "visual_topology_check": "not_deterministically_verified",
            "skill_execution_semantics": "not_deterministically_verified"}


def solve(backend: Any, *, row: dict, prompt: str, skill_count: int,
          model: str, template: str, folder: Path, attempts: int, previous_result: dict | None = None,
          retry_failed: bool = False) -> dict:
    bundle = bundle_for(row, prompt)
    previous: dict = previous_result or {}
    saved = sorted(folder.glob("check_*.json"), key=lambda path: int(path.stem.split("_")[-1]))
    start = 0
    if saved:
        previous = read_json(saved[-1])
        if previous.get("student_prompt") != bundle["text"] or previous["teacher_model"] != model:
            raise ValueError("Cached execution has a different prompt or model")
        start = int(saved[-1].stem.split("_")[-1]) + 1
        if previous["accepted"] or (start >= attempts and not retry_failed):
            return previous
    remaining = attempts - start % attempts
    for attempt in range(start, start + remaining):
        payload = {"sample_id": row["sample_id"], "student_prompt": bundle["text"],
                   "previous_output": previous.get("output"),
                   "feedback": previous.get("verification", {}).get("errors", [])}
        # GT stays in the local verifier, never in this execution request.
        value = backend.request(folder / f"attempt_{attempt:02d}.json", model=model,
                                instruction=template, payload=payload, images=[bundle["image_path"]])
        result = verify(value, row, bundle, skill_count)
        previous = {"sample_id": row["sample_id"], "domain": row["domain"],
                    "teacher_model": model,
                    "output": value.get("output", "") if isinstance(value, dict) else "",
                    "metrics": value.get("metrics") if isinstance(value, dict) else None,
                    "verification": result, "accepted": result["passed"],
                    "image_path": bundle["image_path"], "student_prompt": bundle["text"]}
        write_json(folder / f"check_{attempt:02d}.json", previous)
        if result["passed"]:
            break
    return previous


def example_text(index: int, row: dict, response: dict) -> str:
    bundle = bundle_for(row, "{question}")
    # Literal braces in a source question must survive later template rendering.
    source = (f"Task: {row['question']}\nWeights: {list(_weights_tuple(row))}\n"
              f"Vertex Table:\n{bundle['table_text']}\n").replace("{", "{{").replace("}", "}}")
    output = response["output"].replace("{", "{{").replace("}", "}}")
    return f"## Example {index}\n\n{source}\n{output}"
