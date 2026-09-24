# Third-Party and Data Notice

This repository contains RLHarness project code, prompts, fixed split IDs, and metadata required for reproduction. It does not include base models, final models, API keys, training rollouts, or raw MapTab images and tables.

`third_party/` pins the LLaMA-Factory and VERL source snapshots and records SkillOpt provenance and licensing. Installable SkillOpt source is under `src/skillopt/`. Authoritative licenses are stored in each component directory; provenance and local changes are documented in [`third_party/README.md`](third_party/README.md). Each component remains subject to its own license.

Obtain MapTab from its official repository or Hugging Face page and follow its license and usage terms. If the historical training snapshot cannot be reconstructed from the current public release, users must provide a legally obtained copy. This repository validates only its structure, size, and fixed IDs.

The Qwen3.5 base model and published final weights are distributed separately. This repository does not supersede their licenses or usage restrictions.
