"""Convert route-planning rows to the native modern-verl RL schema."""

from __future__ import annotations

from typing import Any


DATA_SOURCE = "maptab_route"
DEFAULT_IMAGE_MAX_PIXELS = 1_000_000
DEFAULT_IMAGE_MIN_PIXELS = 3_136


def build_verl_row(
    row: dict[str, Any],
    *,
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
    image_min_pixels: int = DEFAULT_IMAGE_MIN_PIXELS,
) -> dict[str, Any]:
    """Build one single-turn multimodal route-planning sample."""
    prompt_text = str(row.get("prompt") or row.get("question") or "").strip()
    if not prompt_text:
        raise ValueError(f"Sample {row.get('sample_id')!r} has an empty prompt.")

    image_paths = [str(path) for path in (row.get("images") or []) if path]
    if len(image_paths) != 1:
        raise ValueError(
            f"Sample {row.get('sample_id')!r} must contain exactly one image, "
            f"got {len(image_paths)}."
        )

    ground_truth = str(row.get("gt_route") or row.get("answer") or "").strip()
    if not ground_truth:
        raise ValueError(f"Sample {row.get('sample_id')!r} has an empty route.")

    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {
                "role": "user",
                "content": f"<image>\n{prompt_text}",
            }
        ],
        "images": [
            {
                "image": image_paths[0],
                "max_pixels": int(image_max_pixels),
                "min_pixels": int(image_min_pixels),
            }
        ],
        "ability": "multimodal_route_planning",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "sample_id": str(row.get("sample_id") or ""),
            "index": int(row.get("index") or 0),
            "domain": str(row.get("domain") or ""),
            # A validated task-specific image prior is not available. Hybrid-DGPO is
            # deliberately driven by rollout pass rate in the first pilot.
            "image_difficulty": 0.0,
            "table_path": str(row.get("table_path") or ""),
            "gt_route": ground_truth,
        },
    }
