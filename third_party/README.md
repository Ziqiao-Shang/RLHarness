# Third-Party Source

This directory pins source required for training. It does not contain Python environments, models, data, API keys, or runtime artifacts.

| Directory | Upstream project | Purpose |
| --- | --- | --- |
| `LlamaFactory/` | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | multimodal SFT and LoRA merging |
| `verl-modern/` | [VERL](https://github.com/volcengine/verl) | four-GPU GRPO/Hybrid-DGPO training |
| `SkillOpt/` | historical SkillOpt integration snapshot at revision `5409cdc` | provenance and license; installable source is under `src/skillopt/` |

Each directory's `LICENSE` file is authoritative. The SkillOpt engine is installed from `src/skillopt/`, while the map-domain adapter is under `src/rlharness/initialization/`. Compatibility and security patches are documented in `SkillOpt/README.md`; this snapshot must not be represented as an unmodified upstream release.
