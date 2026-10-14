# v45: Reset causes and next repair

Task `t135914` finished on node007 GPU0 in 226.45 seconds: 24 independent
normal references, 80 frozen-policy development rollouts, 196,608 interactions,
zero RL updates/replay writes. Only 3,338,449 bytes of scalar diagnostic JSON
were synced. References/model/config/runtime, trajectories/residuals and
episode-centered belief statistics reproduce v44 exactly.

- Fixed-policy normal validation: all 15 major resets occur at age=768.
  Holding the previous calibration group on that same residual removes all 15;
  global-centered full replay reduces the validation count from 15/20 to 0/20.
- Relative-policy persistent heavy: all 20 late64 resets occur at age=256,
  despite no new mode switch. Same-step held-group prediction removes all 20;
  global-centered reduces the late64 count from 20/20 to 0/20.
- Relative-policy normal validation: one reset is an episode-start calibration
  change; four others are isolated spikes at ages 42-44, not boundaries.
  Global-centered still has 4/20 validation cases with major reset.

## Next Step

Remove hard age-calibration switches from the candidate input. The remaining
normal spikes also motivate `scripts/v46_belief.py`: keep learned location,
predictive variance, hazard/history unchanged, but retain Student-t df=4 tails
instead of letting df grow to 23. This moment-matched candidate is implemented;
21 tests pass, including the four observed spike likelihood comparisons and
synthetic isolated-spike versus persistent-shift behavior.

Next replay all four unchanged profiles with the same frozen sources and
normal-only reference splits: episode/global calibration crossed with original/
robust adaptive likelihood. Measure whole-rollout normal false triggers, early
paired lambda response, late recovery and multiple-switch response before RL.

## Limitations

This localizes reused development failures, not independent RL improvement.
The held-group counterfactual changes one observation only. The new robust
likelihood is a predictive approximation and has not yet been tested on full
Hopper trajectories. Full diagnostics: `results/general_ood_v45_reset_localization/summary.json`.

Follow-up: the four-arm v46 replay is complete; see
`docs/general_ood_v46_crossed_tails_results.md`. Multiple-switch testing precedes RL.
