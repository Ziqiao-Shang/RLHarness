# Locked MetroMap Split

The three ID lists come from the historical `nothink_reasoning_subset_v1`: Train1600, Val100, and Test400.

SkillOpt draws a stratified Train480 from Train1600. SFT and RL continue to use the full Train1600.

This historical split is sample-disjoint, not fully map-disjoint: Train1600 and Val100 share 64 maps. Restore it exactly with `data_process.data_split --fixed-ids-dir`; do not resample it.
