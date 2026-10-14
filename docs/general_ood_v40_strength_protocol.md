# v40 strength-matched switch diagnostic

Frozen 2026-10-06. v39 remains HOLD. Its relative-surprise arm averaged beta=-2.29311, barely changed between 1000-step blocks, and lost to fixed-minus2 in ID, paired Markov OOD and worst fixed-profile return. This diagnostic separates penalty magnitude from time-varying belief response; it is not a new benchmark or seed expansion.

## Intervention

Hopper-v4, seed1, three arms on one Linux CUDA runtime:

- `relative-adaptive`: unchanged v39 relative-L1 surprise and corrected age belief, penalty_scale=1.
- `fixed-minus2`: same surprise and corrected belief, penalty_scale=0.
- `fixed-minus2p3`: same surprise with delta age-zero belief (lambda=1), penalty_scale=0.3, yielding beta=-2.3 after warmup.

The -2.3 coefficient is the rounded v39 update-weighted mean, fixed before this run, not selected by an OOD sweep. All arms retain the identical first100 optimizer updates at beta=-2. Actor, context, critic target, BC, entropy, learning rates and replay settings are matched; NAU/SINDy/Jacobian terms remain absent. The fixed-minus2p3 belief is deliberately constant: its reset/window statistics are control construction, not an inferred response. Fixed-minus2 still tracks belief but does not apply it to the actor penalty.

## Measurements

Exactly100000 interactions and99745 optimizer updates per arm. Same independent seed1 mode clock every5000 interactions, including same-mode redraws. Training uses existing v38 mode profiles, not the structured50x evaluation perturbations. Final checkpoint only; saved training context statistics restored for inference.

Retain1000-step training means. Add50-interaction bins from250 interactions before to500 after each noninitial clock tick. Record mode before/after and whether it actually changed, update count, mean reward, episode endings, surprise components/dominance, belief reset/window/entropy, lambda, beta, alpha, Q spread and losses. Retain within-bin minima/maxima for surprise, reset probability, lambda and beta. A tick at zero-based step s occurs before transition s (the s+1-th collected transition); offsets are half-open. Metrics follow the update after that transition. Instrumentation adds no draws, interactions or optimizer calls. Same-mode redraws provide a descriptive control, not a randomized causal contrast. Episode endings help distinguish context resets from profile changes.

The belief consumes surprise from the most recent256-transition rollout for reward/Q spread; the reward normalization variance comes from replay batches, and critic L1 drift uses global critic norms. This is not an isolated new-transition likelihood. Inspect actual-change traces separately from same-mode ticks; compare pre/post means and peak timing without inventing a pass threshold. A response to window/normalizer/optimizer drift alone is not environment identification. Belief is updated in training only; evaluation measures the resulting policy/context behavior, not online belief adaptation to50x OOD.

Evaluation uses20 paired episode seeds starting80000, ID and structured Markov50x period200/probability1, plus5 episodes for each non-ID fixed profile. OOD RNG is explicitly reseeded per episode; each block restores training context. No optimizer/belief updates during evaluation. Report actual non-ID exposure, worst fixed-profile mean, and the paired episode return differences between arms. Episode pairs are not independent training seeds.

## Dispatch And Decision

First run a three-arm6000-interaction CUDA preflight:5745 updates, including the full first switch window at5000. Evaluate2 primary/1 fixed-profile episode. Require all three valid artifacts, matched configs/runtime/mode events, finite metrics, aligned window coverage, fixed coefficients and restored context before formal submission. This is an operational test, not performance evidence.

Formal runs use the same frozen source. Scheduler chooses an idle compatible Linux node without preferred/required placement; never Windows. Only small JSON/log outputs sync locally; checkpoints remain remote. Interruptions restart this short diagnostic.

Compare adaptive with both fixed controls at the same budget. If adaptive does not match or improve both controls on ID, exposed Markov OOD and worst fixed-profile mean, retain HOLD and do not expand seeds. If fixed-minus2p3 reproduces adaptive deterioration, diagnose penalty magnitude before redesigning belief; if it performs differently, inspect timing/trajectory rather than attributing everything to the mean coefficient. Advance only when performance supports adaptive value, switch-aligned traces show a meaningful response to actual profile changes, and alpha/Q/loss do not show sustained escalation. A positive outcome still requires a broader matched-budget screen.

## Limits

One previously selected development seed, short budget, no independent confirmation. Fifty-step bins can conceal sub-bin ordering; extrema retain peaks but not exact peak times. Training profile changes differ from structured50x evaluation. No comparison with v38's300k endpoint as an equal-budget control, and no manuscript claim changes from this diagnostic.

Post-run clarification2026-10-06: the original description incorrectly called all surprise replay-batch based. Source inspection confirmed the recent-rollout path is enabled and supplied on every training update. Corrected above; no intervention, budget, evaluation protocol, decision condition or frozen runtime source changed.
