# Configuration Files

MetroMap and TravelMap share five configuration files:

```text
skillopt.yaml       initial SkillOpt settings
sft_train.yaml      LLaMA-Factory SFT settings
sft_merge.yaml      LoRA merge settings
rl_round1.yaml      first-round GRPO settings
rl_round2.yaml      second-round GRPO continuation settings
```

`rl_round2.yaml` inherits the first-round settings through `_base_config: rl_round1.yaml`. `rlharness.rl.rl_launcher` resolves this inheritance.

Domain differences are not duplicated in separate YAML files. Public scripts inject data paths, prompts, experiment names, checkpoints, and output directories from `--domain metromap|travelmap`.

Model, data, prompt, checkpoint, and output paths can be overridden with environment variables. See [`../docs/training.md`](../docs/training.md).
