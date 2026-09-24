#!/usr/bin/env python3
"""Merge a project override YAML over verl's PPO config and launch training."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def batch_geometry(config) -> dict[str, int]:
    """Resolve verl's prompt-level mini-batch setting into trajectories."""
    prompt_count = int(config.data.train_batch_size)
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    mini_prompt_count = int(config.actor_rollout_ref.actor.ppo_mini_batch_size)
    ppo_epochs = int(config.actor_rollout_ref.actor.ppo_epochs)
    trajectory_count = prompt_count * rollout_n
    mini_trajectory_count = mini_prompt_count * rollout_n
    if mini_prompt_count <= 0 or prompt_count % mini_prompt_count != 0:
        raise ValueError(
            "data.train_batch_size must be divisible by actor.ppo_mini_batch_size "
            f"in prompt units, got {prompt_count} and {mini_prompt_count}"
        )
    return {
        "prompt_count": prompt_count,
        "rollout_n": rollout_n,
        "trajectory_count": trajectory_count,
        "mini_prompt_count": mini_prompt_count,
        "mini_trajectory_count": mini_trajectory_count,
        "optimizer_updates": prompt_count // mini_prompt_count * ppo_epochs,
    }


def print_batch_geometry(config) -> None:
    geometry = batch_geometry(config)
    print(
        "[batch geometry] "
        f"prompts={geometry['prompt_count']}; "
        f"rollout_n={geometry['rollout_n']}; "
        f"trajectories={geometry['trajectory_count']}; "
        f"ppo_mini_batch={geometry['mini_prompt_count']} prompts/"
        f"{geometry['mini_trajectory_count']} trajectories; "
        f"optimizer_updates_per_step={geometry['optimizer_updates']}",
        flush=True,
    )


def validate_lora_reference(config) -> None:
    """Prevent KL from silently using the pre-SFT base as reference."""
    model = config.actor_rollout_ref.model
    actor = config.actor_rollout_ref.actor
    adapter_path = model.get("lora_adapter_path")
    if not adapter_path or not bool(actor.get("use_kl_loss", False)):
        return
    if os.environ.get("ALLOW_LORA_BASE_REFERENCE") == "1":
        print(
            "[reference warning] Loaded LoRA will be disabled for reference log-probs; "
            "KL therefore targets the model underneath that adapter.",
            flush=True,
        )
        return
    raise ValueError(
        "KL with model.lora_adapter_path would use the model underneath the loaded "
        "adapter as reference, not the loaded SFT policy. Merge the SFT adapter into "
        "the base model and train a fresh RL LoRA. Set ALLOW_LORA_BASE_REFERENCE=1 "
        "only when the pre-adapter base is deliberately the reference."
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args, overrides = parser.parse_known_args()
    return args, [item.lstrip("+") for item in overrides if item != "--"]


def load_project_config(path: Path, omega_conf, seen: set[Path] | None = None):
    """Load a project YAML with an optional relative ``_base_config``."""
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic _base_config reference: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing project config: {path}")
    seen.add(path)
    current = omega_conf.load(path)
    base_value = current.pop("_base_config", None)
    if not base_value:
        return current
    base_path = Path(str(base_value))
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = load_project_config(base_path, omega_conf, seen)
    return omega_conf.merge(base, current)


def main() -> None:
    args, overrides = parse_args()
    config_path = args.config.resolve()
    verl_root = args.verl_root.resolve()
    if not (verl_root / "verl/trainer/config/ppo_trainer.yaml").is_file():
        raise FileNotFoundError(f"Not a modern verl checkout: {verl_root}")

    sys.path.insert(0, str(verl_root))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = str((verl_root / "verl/trainer/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, load_project_config(config_path, OmegaConf))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))

    if args.print_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        return

    # Register project-local advantage estimators before the trainer resolves
    # algorithm.adv_estimator through verl's runtime registry.
    from rlharness.common.config import ROOT

    sys.path.insert(0, str(ROOT / "src"))
    adv_estimator_module = config.algorithm.get("adv_estimator_module")
    if adv_estimator_module:
        importlib.import_module(adv_estimator_module)

    from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    validate_lora_reference(config)
    print_batch_geometry(config)
    if args.validate_only:
        print("[config validation] passed", flush=True)
        return
    if not config.trainer.use_v1:
        raise ValueError("RLHarness requires trainer.use_v1=true for this pinned verl version.")
    run_ppo(config, task_runner_class=TaskRunnerV1)


if __name__ == "__main__":
    main()
