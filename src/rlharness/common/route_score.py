"""MapTab route scoring (all_acc / part_acc). Adapted from MapTab eval, no tools."""

from __future__ import annotations

import difflib
from typing import Sequence


def _normalize_station(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "").replace("_", "")


def split_route(route: str) -> list[str]:
    if not route:
        return []
    # Keep (transfer) glued to station token as in MapTab GT
    return [p.strip() for p in route.strip().split("-") if p.strip()]


def stations_match(a: str, b: str, threshold: float = 0.5) -> bool:
    na, nb = _normalize_station(a), _normalize_station(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


def prefix_match_len(pred: Sequence[str], gt: Sequence[str], threshold: float = 0.5) -> int:
    n = 0
    for p, g in zip(pred, gt):
        if stations_match(p, g, threshold=threshold):
            n += 1
        else:
            break
    return n


def score_route(pred_route: str, gt_route: str, threshold: float = 0.5) -> dict[str, float]:
    pred = split_route(pred_route)
    gt = split_route(gt_route)
    if not gt:
        return {"all_acc": 0.0, "part_acc": 0.0, "pred_len": float(len(pred)), "gt_len": 0.0}
    if len(pred) == len(gt) and all(
        stations_match(p, g, threshold=threshold) for p, g in zip(pred, gt)
    ):
        return {
            "all_acc": 1.0,
            "part_acc": 1.0,
            "pred_len": float(len(pred)),
            "gt_len": float(len(gt)),
        }
    pref = prefix_match_len(pred, gt, threshold=threshold)
    if len(gt) == 1:
        part = float(pref == 1)
    else:
        # The start station is given in the question. Reward only correctly
        # predicted route edges, not merely repeating that station.
        part = max(pref - 1, 0) / (len(gt) - 1)
    return {
        "all_acc": 0.0,
        "part_acc": float(part),
        "pred_len": float(len(pred)),
        "gt_len": float(len(gt)),
    }
