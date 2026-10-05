# Nonlinear load v21: first-action objective diagnostic

t128459/t128460 exited0 in52.8/53.0 seconds on scheduler-selected Linux GPUs. Both source actors reproduced all saved v20 costs exactly; external paths matched and parameters stayed unchanged. There were2,400 controller episodes, zero optimizer updates and zero fitting interactions. Only1,523,343 bytes of formal result JSON returned; checkpoints/replay stayed remote. Five focused tests pass, including JSON round-trip validation with native float32 scores.

## Findings

At radius1, replacing only the first action with the own fitted-reference action lowers both raw and gamma0.99-discounted mean cost for both seeds/loads. Subsequent actions use the frozen actor. Deltas below are fitted-first minus actor; lower cost and higher score are preferred. Every block uses100 paired paths.

| Seed | Load | Raw Cost Delta | Discounted Cost Delta | Q Delta | Risk Delta | Soft Surrogate Delta |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 1 | -0.021857 | -0.018958 | +0.000478 | +0.000804 | -0.000564 |
| 16 | 1 | -0.026728 | -0.023042 | -0.000668 | -0.000724 | -0.002037 |
| 15 | 50 | -0.021955 | -0.018964 | +0.001239 | +0.002020 | +0.000620 |
| 16 | 50 | -0.026431 | -0.022762 | +0.002970 | +0.005833 | +0.004463 |

The raw-cost reductions are0.425%/0.292% for seed15 ID/OOD and0.516%/0.350% for seed16. State-cost improvement occurs almost entirely in the first50 steps, offset partly by higher command cost. Discount changes only1/0 and3/2 individual path signs in these radius1 ID/OOD blocks, without changing their mean ordering. It does reverse the mean ordering at seed16 radius0.2/ID; the tiny raw delta+1.46e-5 becomes discounted-3.82e-6, with36 path reversals. Both radius0.2/OOD mean costs worsen under the first-action replacement.

At radius1/ID, discounted cost favors fitted-first on92/87 paths for seed15/16. The risk score nevertheless prefers the actor on23/48 of those paths; the pointwise soft surrogate does so on67/59. Entropy reverses the mean risk preference in seed15 and reinforces the pre-existing mismatch in seed16. Both checkpoints have alpha at0.01, while terminal reward standard deviations are33.6514/23.1168. Thus entropy/reward scale is a plausible intervention target, but entropy alone does not explain both seeds' rankings.

## Next Step

Retain full Jacobian, cubic-local reference and the current environment. Test only `enable_entropy_floor=false` against the unchanged floor0.01 arm on fresh matched seeds17/18. Keep automatic temperature learning, reward EMA, risk/belief updates, sampling, resets, identification, budgets and last-checkpoint selection fixed. Use the existing paired radii/loads and admission gates: radius1 total/early-state improvement on each seed/load, radius0.2 total-cost retention, and all-radius own-reference added value. No critic repair, reward-scale change or seed expansion in the same experiment. This four-task training comparison was not dispatched in v21; it is now implemented as v22 with tasks t128677--t128680. See `docs/nonlinear_load_v22_entropy_floor_protocol.md` for the frozen protocol and verified dispatch.

## Limitations

The pointwise entropy surrogate is not the expected stochastic actor objective. Q describes reward-normalized soft stochastic continuation, whereas this diagnostic measures deterministic raw continuation; disagreement is not a critic-error certificate. Terminal alpha/reward scale does not reconstruct their training history. This first-action diagnostic neither explains the full policy/reference gap nor establishes general RL or learner-comparator superiority. Historical HOLD decisions and manuscript tables remain unchanged.

Evidence: `results/nonlinear_load_v21_objective_diagnostic/{task_observations,summary}.json`, the two per-seed JSONs and the frozen inputs/manifest in `experiments/nonlinear_load_v21_objective_20261003`. Reproduce with `python3 scripts/summarize_nonlinear_load_v21.py` using the documented frozen package versions.
