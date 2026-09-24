"""Self-contained Hybrid-DGPO advantage estimator for RLHarness."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


ADV_ESTIMATOR_NAME = "maptab_hybrid_dgpo"


@register_adv_est(ADV_ESTIMATOR_NAME)
def compute_hybrid_dgpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config=None,
    batch=None,
    epsilon: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center group rewards and apply positive question-level difficulty weights."""
    del kwargs
    if batch is None or "accuracy_rewards" not in batch or "image_difficulties" not in batch:
        raise ValueError(
            "Hybrid DGPO requires accuracy_rewards and image_difficulties from reward_extra_info"
        )

    settings = config.get("hybrid_dgpo", {}) if config is not None else {}
    temperature = float(settings.get("difficulty_temperature", 2.0))
    image_prior_strength = float(settings.get("image_prior_strength", 1.0))
    enable_question_weighting = bool(settings.get("enable_question_weighting", True))
    if temperature <= 0:
        raise ValueError("hybrid_dgpo.difficulty_temperature must be positive")
    if image_prior_strength < 0:
        raise ValueError("hybrid_dgpo.image_prior_strength must be non-negative")

    scores = token_level_rewards.sum(dim=-1)
    accuracy = batch["accuracy_rewards"].to(device=scores.device, dtype=scores.dtype)
    image_difficulty = batch["image_difficulties"].to(
        device=scores.device, dtype=scores.dtype
    )
    if accuracy.ndim != 1 or image_difficulty.ndim != 1:
        raise ValueError("Hybrid DGPO scalar metadata must have shape [batch]")
    if not torch.all((accuracy >= 0) & (accuracy <= 1)):
        raise ValueError("accuracy_rewards must be in [0, 1]")
    if not torch.all((image_difficulty >= 0) & (image_difficulty <= 1)):
        raise ValueError("image_difficulties must be in [0, 1]")

    groups: OrderedDict[object, list[int]] = OrderedDict()
    for row, uid in enumerate(index):
        groups.setdefault(uid, []).append(row)

    centered = torch.zeros_like(scores)
    pass_rates = torch.zeros_like(scores)
    dynamic_difficulties = torch.zeros_like(scores)
    hybrid_difficulties = torch.zeros_like(scores)
    question_weights = torch.zeros_like(scores)
    variance_sources = {
        "final": torch.zeros_like(scores),
        "exact": torch.zeros_like(scores),
        "partial": torch.zeros_like(scores),
        "format": torch.zeros_like(scores),
        "overlong": torch.zeros_like(scores),
    }
    component_keys = {
        "exact": "accuracy_rewards",
        "partial": "part_acc_rewards",
        "format": "format_rewards",
        "overlong": "overlong_rewards",
    }
    valid_groups: list[tuple[list[int], torch.Tensor]] = []

    with torch.no_grad():
        for uid, rows in groups.items():
            row_idx = torch.as_tensor(rows, device=scores.device, dtype=torch.long)
            group_scores = scores[row_idx]
            group_accuracy = accuracy[row_idx]
            group_images = image_difficulty[row_idx]
            if torch.max(group_images) - torch.min(group_images) > epsilon:
                raise ValueError(f"Inconsistent image difficulty within prompt group {uid}")

            score_mean = group_scores.mean()
            mean_absolute_deviation = torch.mean(torch.abs(group_scores - score_mean))
            pass_rate = group_accuracy.mean()
            dynamic = 1.0 - pass_rate
            group_size = float(len(rows))
            hybrid = (
                group_size * dynamic + image_prior_strength * group_images.mean()
            ) / (group_size + image_prior_strength)

            pass_rates[row_idx] = pass_rate
            dynamic_difficulties[row_idx] = dynamic
            hybrid_difficulties[row_idx] = hybrid
            if float(mean_absolute_deviation) > epsilon:
                centered[row_idx] = group_scores - score_mean
                valid_groups.append((rows, hybrid))
                variance_sources["final"][row_idx] = 1.0
            for source, key in component_keys.items():
                if key not in batch:
                    continue
                values = batch[key][row_idx]
                if float(torch.max(values) - torch.min(values)) > epsilon:
                    variance_sources[source][row_idx] = 1.0

        if valid_groups:
            if enable_question_weighting:
                logits = torch.stack([difficulty for _, difficulty in valid_groups]) / temperature
                weights = len(valid_groups) * torch.softmax(logits, dim=0)
            else:
                weights = torch.ones(
                    len(valid_groups), device=scores.device, dtype=scores.dtype
                )
            for (rows, _), weight in zip(valid_groups, weights, strict=True):
                row_idx = torch.as_tensor(rows, device=scores.device, dtype=torch.long)
                question_weights[row_idx] = weight

        weighted_scores = centered * question_weights
        advantages = weighted_scores.unsqueeze(-1) * response_mask
        batch["hybrid_pass_rate"] = pass_rates
        batch["hybrid_dynamic_difficulty"] = dynamic_difficulties
        batch["hybrid_difficulty"] = hybrid_difficulties
        batch["hybrid_question_weight"] = question_weights
        for source, values in variance_sources.items():
            batch[f"hybrid_{source}_variance"] = values

    return advantages, advantages
