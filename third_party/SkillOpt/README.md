# SkillOpt Provenance

Source: revision `5409cdc` of the project's historical SkillOpt integration repository, dated 2026-09-11. See `LICENSE` for the MIT license.

Installable source is under `src/skillopt/`. This directory retains only provenance information and the upstream license; it contains no models, trajectories, secrets, or environment data.

The main loop remains the original `ReflACTTrainer`: rollout, reflection, merge, ranking, patching, and validation gating. Domain adapters and entry points are under `src/rlharness/initialization/skillopt_*.py`. MetroMap uses the analysis prompts included with this snapshot. TravelMap uses generic analysis prompts and supplies its own task rules in the trajectory instead of applying MetroMap-specific scoring rules.

Local compatibility and security patches fully redact configured secrets, prevent API errors from echoing response bodies, avoid passing GPT-only `reasoning_effort` parameters to the Gemini student, and preserve zero scores and completed-step reflection summaries during resume.

This is not a byte-for-byte reproduction of the historical experiment settings or results. The current protocol uses Train480, a Gemini-3.5 student, a GPT-5.6 teacher, and no test-set evaluation during optimization.
