# Verified Runtime Environments

RLHarness separates lightweight orchestration, SFT, and GRPO/evaluation into three Python environments. The root `requirements.txt` installs only data processing, API, and orchestration dependencies; it does not install the GPU training stacks.

The combinations below were used to import, regression-test, or execute the current training path on 2026-09-24. Other versions may work, but they are outside the verified scope.

## Hardware and System

| Item | Verified configuration |
| --- | --- |
| OS kernel | Linux 5.15.0-78-generic, x86_64 |
| GPU | 4 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB |
| NVIDIA driver | 595.71.05 |
| Driver-supported CUDA | 13.2 |
| PyTorch CUDA build | 13.0 |

## Lightweight Orchestration and SkillOpt

Used by `00_initialize_skills.sh`, `01_generate_sft_data.sh`, skill evolution, and CPU tests.

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| OpenAI Python SDK | 3.8.0 |
| NumPy | 2.5.3 |
| PyYAML | 6.0.3 |
| json-repair | 0.63.4 |
| Tokenizers | 0.22.2 |
| SkillOpt | vendored under `src/skillopt/`, provenance revision `5409cdc` |

Install with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## SFT Environment

Used by `02_train_sft.sh` and LoRA merging.

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.11.0 |
| PyTorch CUDA | 13.0 |
| Transformers | 5.8.0 |
| LLaMA-Factory | 0.9.6.dev0 from `third_party/LlamaFactory/` |
| Accelerate | 1.11.0 |
| PEFT | 0.18.1 |
| Triton | 3.6.0 |
| NumPy | 2.5.2 |
| Pillow | 11.3.0 |
| PyYAML | 6.0.3 |

Pass the environment entry points explicitly:

```bash
export PYTHON_BIN=/path/to/sft-env/bin/python
export LLAMAFACTORY_CLI=/path/to/sft-env/bin/llamafactory-cli
```

## GRPO and Local Evaluation Environment

Used by `03_train_rl_round1.sh`, `05_train_rl_round2.sh`, and `06_evaluate_test.sh`.

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.11.0 |
| PyTorch CUDA | 13.0 |
| Transformers | 5.10.0 |
| vLLM | 0.26.0 |
| Ray | 2.56.1 |
| VERL | 0.9.0.dev0 from `third_party/verl-modern/` |
| Accelerate | 1.14.0 |
| PEFT | 0.20.0 |
| Triton | 3.6.0 |
| Hydra Core | 1.3.5 |
| OmegaConf | 2.3.1 |
| NumPy | 2.3.5 |
| Pillow | 12.3.0 |

Use the same environment for training and vLLM evaluation:

```bash
export VERL_PYTHON=/path/to/verl-env/bin/python
export VLLM_PYTHON=/path/to/verl-env/bin/python
```

`eval_local` selects native vLLM for Qwen3.5 when vLLM 0.26 or newer is
installed. For older local vLLM versions, merged checkpoints fall back to
direct Transformers generation with batch size one. Set
`RLHARNESS_EVAL_BACKEND=vllm` or `transformers` to override this choice;
the direct Transformers path requires Accelerate and does not load LoRA
adapters.

## Environment Boundaries

- Do not force LLaMA-Factory and VERL/vLLM into one environment. Their verified Transformers, PEFT, and Accelerate versions differ.
- `third_party/LlamaFactory/` and `third_party/verl-modern/` are the source trees called by this project. Installing similarly named PyPI packages does not guarantee equivalent code.
- The CUDA toolkit, driver, and PyTorch build must be compatible. The table records the verified combination, not a requirement to use the same GPU model.
- Hosted API models may change server-side. Formal runs should record model routes, execution time, prompt files, and raw responses.
