"""Self-contained route-planning reward for RLHarness and verl."""

from __future__ import annotations

import difflib
import re


REASONING_RESPONSE_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>\s*<response>(.*?)</response>",
    re.DOTALL | re.IGNORECASE,
)
RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _normalize_station(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "").replace("_", "")


def _split_route(route: str) -> list[str]:
    if not route:
        return []
    return [p.strip() for p in route.strip().split("-") if p.strip()]


def _stations_match(a: str, b: str, threshold: float = 0.9) -> bool:
    na, nb = _normalize_station(a), _normalize_station(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


def _part_acc(pred_route: str, gt_route: str, threshold: float) -> float:
    pred = _split_route(pred_route)
    gt = _split_route(gt_route)
    if not gt:
        return 0.0
    if len(pred) == len(gt) and all(
        _stations_match(p, g, threshold) for p, g in zip(pred, gt)
    ):
        return 1.0
    n = 0
    for p, g in zip(pred, gt):
        if _stations_match(p, g, threshold):
            n += 1
        else:
            break
    if len(gt) == 1:
        return float(n == 1)
    return max(n - 1, 0) / (len(gt) - 1)


def _all_acc(pred_route: str, gt_route: str, threshold: float) -> float:
    pred = _split_route(pred_route)
    gt = _split_route(gt_route)
    if not gt:
        return 0.0
    if len(pred) == len(gt) and all(
        _stations_match(p, g, threshold) for p, g in zip(pred, gt)
    ):
        return 1.0
    return 0.0


def _parse(predict_str: str) -> tuple[bool, str]:
    text = (predict_str or "").strip()
    match = REASONING_RESPONSE_RE.fullmatch(text)
    resp = RESPONSE_RE.search(text)
    pred = resp.group(1).strip() if resp else ""
    format_ok = bool(
        match
        and match.group(1).strip()
        and match.group(2).strip()
        and not THINK_RE.search(text)
    )
    return format_ok, pred


def _score(predict_str: str, ground_truth, *, extra_info=None, **kwargs) -> dict[str, float]:
    values = dict(kwargs)
    if isinstance(extra_info, dict):
        for key in (
            "format_weight",
            "part_acc_weight",
            "all_acc_weight",
            "require_skill_retrieve",
            "station_match_threshold",
            "image_difficulty",
        ):
            if key in extra_info:
                values[key] = extra_info[key]

    format_weight = float(values.get("format_weight", 0.05))
    part_w = float(values.get("part_acc_weight", 0.25))
    all_w = float(values.get("all_acc_weight", 0.70))
    threshold = float(values.get("station_match_threshold", 0.90))
    image_difficulty = float(values.get("image_difficulty", 0.0))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"station_match_threshold must be in [0, 1], got {threshold}")
    if not 0.0 <= image_difficulty <= 1.0:
        raise ValueError(f"image_difficulty must be in [0, 1], got {image_difficulty}")
    fmt_ok, pred = _parse(predict_str)
    gt = str(ground_truth or "")
    part = _part_acc(pred, gt, threshold)
    all_acc = _all_acc(pred, gt, threshold)
    reward = float(
        format_weight * (1.0 if fmt_ok else 0.0) + part_w * part + all_w * all_acc
    )
    return {
        "score": reward,
        "accuracy_reward": float(all_acc),
        "part_acc_reward": float(part),
        "format_reward": 1.0 if fmt_ok else 0.0,
        "image_difficulty": image_difficulty,
    }


def maptab_route_compute_score(predict_str: str, ground_truth, **kwargs) -> float:
    """Compatibility scalar API for the legacy RewardMap checkout."""
    return _score(predict_str, ground_truth, **kwargs)["score"]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> dict[str, float]:
    """Modern verl API: 0.05 format + 0.25 prefix + 0.70 exact route."""
    del data_source
    return _score(
        solution_str,
        ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
