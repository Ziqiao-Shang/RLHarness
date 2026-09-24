# RLHarness Reward Design

This document specifies the reward used by the two MetroMap GRPO rounds in RLHarness. It follows the repository code and configuration rather than inferring behavior from metric names in logs.

Scope:

- First GRPO round: train the SFT model through `global_step_400`.
- Second GRPO round: continue from `global_step_400` through `global_step_600` with the evolved prompt and skill bank.
- Both rounds use the same route reward, length penalty, group filtering, and Hybrid-DGPO advantage estimator.

Key implementation files:

- Route reward: `src/rlharness/rl/reward.py`
- Hybrid-DGPO: `src/rlharness/rl/advantage.py`
- DAPO reward manager: `third_party/verl-modern/verl/workers/reward_manager/dapo.py`
- First-round configuration: `configs/rl_round1.yaml`
- Second-round configuration: `configs/rl_round2.yaml`

Both domains share reward and training configurations. Launch scripts inject domain-specific data paths, prompts, and experiment names from `--domain`.

## 1. Per-Trajectory Task Reward

Each model output receives three components: format reward, partial-route reward, and exact-route reward.

Definitions:

- `F`: format validity, either 0 or 1.
- `P`: accuracy of the contiguous route prefix starting at the origin, in `[0, 1]`.
- `A`: exact correctness of the entire route, either 0 or 1.

The task reward is:

```text
R_task = 0.05 * F + 0.25 * P + 0.70 * A
```

| Component | Weight | Purpose |
| --- | ---: | --- |
| Format `F` | 0.05 | Keep outputs parseable |
| Partial route `P` | 0.25 | Provide dense feedback for a correct contiguous prefix |
| Exact route `A` | 0.70 | Primary objective: the full route must match GT |

`R_task` is in `[0, 1]`.

## 2. Format Reward F

After trimming surrounding whitespace, a valid output must match this structure:

```text
<reasoning>nonempty reasoning</reasoning><response>nonempty route</response>
```

Whitespace is allowed between `</reasoning>` and `<response>`, but all conditions below must hold:

1. Both sections exist and are nonempty.
2. No text appears outside the tags.
3. The output does not contain `<think>...</think>`.

If valid, `F=1`; otherwise `F=0`.

Invalid format does not automatically zero the whole trajectory. If the implementation can still extract a route from `<response>...</response>`, that route can receive partial or exact correctness. A fully correct route with invalid outer formatting can theoretically receive:

```text
0.25 + 0.70 = 0.95
```

If no `<response>` can be parsed, the predicted route is empty and normally receives zero.

## 3. Station Matching

The route is split into station tokens on hyphens. Before comparison, each station name is:

1. Trimmed.
2. Lowercased.
3. Stripped of spaces and underscores.

Two station names match if either condition holds:

- Their normalized forms are identical.
- Their `SequenceMatcher` similarity is at least `0.90`.

This permits minor spelling, spacing, or underscore differences. It does not ignore station order, route length, or transfer markers.

## 4. Exact-Route Reward A

`A=1` only when both conditions hold:

1. Predicted and GT routes have the same number of stations.
2. The predicted station at every position matches the corresponding GT station.

Otherwise, `A=0`. Missing or extra stations, incorrect order, a wrong branch, or mismatched transfer markers can therefore zero the exact reward.

## 5. Partial-Route Reward P

Partial accuracy rewards only the contiguous correct prefix from the origin through the first error. A station that happens to match after an earlier error receives no credit.

Let the GT route contain `M` stations, and let `n` stations match contiguously from the origin. If the route is not already exact:

```text
P = max(n - 1, 0) / (M - 1)
```

Subtracting one converts station count to traversed edge count. The metric is the fraction of GT route edges followed correctly from the origin.

For example, if GT has five stations and the first three predicted stations match before an error at position four:

```text
P = (3 - 1) / (5 - 1) = 0.5
```

With valid formatting and a non-exact route:

```text
R_task = 0.05 + 0.25 * 0.5 = 0.175
```

Matching only the origin gives `P=0`; a full match gives `P=1`.

## 6. Length Penalty

The maximum generation length is 4,096 tokens, with the final 512 tokens reserved as the overlong buffer:

```text
safe length = 4096 - 512 = 3584 tokens
```

Responses of at most 3,584 tokens receive no length penalty. Longer responses receive a linear penalty:

```text
R_len = min(-(L - 3584) / 512 * 0.1, 0)
```

`L` is the number of valid response tokens.

| Response length | Length penalty |
| ---: | ---: |
| `L <= 3584` | 0 |
| `L = 3840` | -0.05 |
| `L = 4096` | -0.10 |

The scalar used by training and group filtering is:

```text
R_final = R_task + R_len
```

Its theoretical range is approximately `[-0.1, 1.0]`.

## 7. Common Reward Examples

The examples below omit unusual parsing edge cases:

| Case | `F` | `P` | `A` | `R_task` |
| --- | ---: | ---: | ---: | ---: |
| Valid format and exact route | 1 | 1 | 1 | 1.000 |
| Exact route with invalid format | 0 | 1 | 1 | 0.950 |
| Valid format and contiguous-prefix accuracy 0.5 | 1 | 0.5 | 0 | 0.175 |
| Valid format but wrong from the first edge | 1 | 0 | 0 | 0.050 |
| Unparseable route and invalid format | 0 | 0 | 0 | 0.000 |

The length penalty is added afterward. For example, an exact output of exactly 4,096 tokens receives `1.0 - 0.1 = 0.9`.

## 8. Eight Rollouts per Question and Group Filtering

Training uses `rollout.n=8`, producing eight candidate trajectories per question. DAPO group filtering uses `final_reward`, not exact correctness alone.

If all eight trajectories for a question have the same `R_final`, the group has no within-group reward variation and is filtered out before another question is sampled. A group is retained whenever final rewards differ because of any component:

- Exact-route reward.
- Partial-route reward.
- Format reward.
- Length penalty.

Consequently, eight incorrect routes are not necessarily filtered: different correct-prefix lengths can still provide training signal. Conversely, eight exact responses with identical format and length are filtered because their relative advantages are all zero.

## 9. Hybrid-DGPO Advantage Estimation

For each retained question, compute the mean final reward of its eight trajectories:

```text
mean_q = mean(R_final_qi)
```

The base relative advantage is:

```text
centered_qi = R_final_qi - mean_q
```

Within a question, trajectories above the mean receive positive advantages and trajectories below the mean receive negative advantages. Standard-deviation normalization is disabled:

```text
norm_adv_by_std_in_grpo: false
```

The actual reward differences are therefore retained rather than divided by the within-group standard deviation.

### 9.1 Dynamic Question Difficulty

Question difficulty is computed only from exact correctness across the eight trajectories:

```text
pass_rate_q = mean(A_qi)
difficulty_q = 1 - pass_rate_q
```

The current experiment uses `1 - pass_rate` directly as difficulty.

This choice follows the motivation of hard-example mining and Focal Loss: reduce the relative influence of examples the current model already solves reliably and allocate more weight to examples it finds difficult. Focal Loss uses a factor such as `(1-p)^gamma`, assigning less weight as confidence in the correct class increases.

Here, the same general idea is lifted from individual classification examples to question-level online outcomes. `pass_rate_q` estimates the current policy's success probability on question `q`, and `1-pass_rate_q` estimates its difficulty. See [Focal Loss](https://arxiv.org/abs/1708.02002) and [DAPO Dynamic Sampling](https://dapo-sia.github.io/).

This should be described precisely as a difficulty-aware reweighting heuristic inspired by hard-example mining, not as the unique optimal weighting derived from GRPO theory. The exact choices of `1-pass_rate`, softmax, and temperature 2.0 are experimental design decisions. Their effect should ultimately be checked with an ablation that disables difficulty weighting.

### 9.2 Difficult-Question Weight

For questions in the current batch that retain valid reward variation, assign weights with a temperature-2.0 softmax:

```text
w_q = N_valid * softmax(difficulty_q / 2.0)
```

`N_valid` is the number of valid question groups, so the mean weight across valid groups remains 1. Questions with lower exact pass rates receive slightly higher weights, while temperature 2.0 limits excessive amplification.

The final advantage is:

```text
Adv_qi = (R_final_qi - mean_q) * w_q
```

This value is broadcast over valid tokens in the corresponding response for the PPO/GRPO actor update.

## 10. Reward Versus Other Training Constraints

The following mechanisms are not part of the route reward formula.

### 10.1 KL Regularization

Configuration:

```text
use_kl_in_reward: false
use_kl_loss: true
kl_loss_coef: 0.001
```

KL is not added to `R_final`. It is a separate actor-loss regularizer that limits drift from the reference model.

### 10.2 Skill Usage

The reward function provides no direct reward for skill markers, invocation count, or explicit use of a particular skill:

```text
require_skill_retrieve: false
```

This field does not enter the current scoring formula. Skills can improve reward only indirectly by improving route correctness, format, or response length. A rising training reward alone therefore does not prove that the model is using the skill bank. Skill usage must be measured separately from trajectories or explicit markers.

## 11. Interpreting Log Metrics

| Metric | Meaning |
| --- | --- |
| `score` | `R_task` before the length penalty |
| `final_reward` | `R_final` after the length penalty |
| `accuracy_reward` | exact indicator `A`; its mean is exact route accuracy |
| `part_acc_reward` | contiguous-origin-prefix accuracy `P` |
| `format_reward` | format indicator `F` |
| `overlong_reward` | length penalty `R_len` |
| `critic/rewards/mean` | mean final reward for the training batch; without KL-in-reward it normally follows final-score statistics |
| `hybrid_pass_rate` | exact success fraction among eight rollouts for a question |
| `hybrid_dynamic_difficulty` | `1 - hybrid_pass_rate` |
| `hybrid_question_weight` | question weight after the difficulty softmax |

The exact accuracy reported on fixed Test400 is the mean of `A`, not the composite `R_final`. Because the composite also includes partial route quality, formatting, and length, training reward can rise without an immediate Test400 exact-accuracy increase, and exact accuracy can rise while short-term composite reward fluctuates.

## 12. Design Summary

The reward is designed to:

1. Make a completely correct route the primary objective with 70% of task-reward weight.
2. Reduce sparse 0/1 feedback with a 25% contiguous-prefix reward.
3. Preserve parseable outputs with a 5% format reward without allowing format to dominate route quality.
4. Reduce excessive length and truncation risk with a penalty of at most 0.1.
5. Use `final_reward` variance to retain all-wrong groups whose partial quality differs.
6. Build relative advantages by centering within each question and mildly upweight questions with low pass rates.

The reward directly supervises final route quality. It does not directly supervise visual recognition, arithmetic reasoning, or the process of using skills. Those processes receive only indirect feedback when they change the final output.
