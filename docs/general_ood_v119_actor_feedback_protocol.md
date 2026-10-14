# v119 frozen actor-feedback pilot

Question: does the v118 recognition gain improve closed-loop performance when
the only intervention remains the scalar actor uncertainty penalty?

Freeze v115 neural mean, v116 matched classifier, v117 normal calibration and
v98 diverse-recipe incumbent. No refitting, gain/threshold tuning or new symbolic
module. Actor/replay/context/config use the same frozen continuation code as v118.

- Environment seeds 135270/271/272, smoke135269; one source policy (seed1,
  update99745), shared fit-only replay6144 and shared context100000.
- Four arms: matched-classifier, v98 suffix-disagreement, fixed-minus2,
  fixed-minus3. Adaptive beta=-2-risk without gain tuning; constants=-2 and -3.
- Each seed/arm starts five independent 6000-step continuations: normal, heavy,
  slippery, high-gravity and compound. Change at3800, all update every real step.
  Total60 paths, 360000 real transitions/optimizer updates; no shadow calls.
- Readout scores only the executed real transition, before its optimizer update.
  It cannot change the action that produced that transition. It receives neither
  the true mode nor the switch time. Filter and departure history persist across
  episode reset; episode age resets. Constants do not require score inference.
- Within an arm, normal/changed trajectories match through the switch action.
  Across arms, only initial state/action/transition are equal; after feedback,
  states, resets and replay contents can diverge. Equal budget is not equal data.
- Primary: paired seed-level difference in post-switch fixed-budget mean reward,
  equally averaged over four changed modes. Report each mode/seed and the normal
  control, including terminations and risk/beta diagnostics. Averages of completed
  episode returns alone are not the primary metric because of censoring.
- Secondary: saved final actor, deterministic three episodes in its training
  mode and normal (normal-only training needs only normal evaluation), max1000
  steps. Evaluation starts140000+1000*seed_index+100*mode_index, paired across
  arms. Context is restored from the final training snapshot before each episode
  and updated within the episode; no optimizer updates or risk feedback at eval.
- Twelve scheduler shards (seed x arm), no preferred GPU/node or Windows nodes.
  Inputs stay on node007, where source/reference checkpoints already reside.
  Only scalar JSON is synced; native traces/checkpoints support exact resume.

Native smoke uses all20 cells, 64steps/switch32 and one max64-step evaluation
episode. It checks online scores against the frozen replay implementation,
recursive/risk history against causal offline replay, matching first transitions
and within-arm prefixes. Local tests additionally interrupt/resume an adaptive
path across native resets and check the actor-only beta slot.

Summarize matched versus v98 and both constant controls using paired environment
seed differences and 95% Student-t intervals (n=3). Do not select a best checkpoint
or drop an adverse seed. No automatic adoption: assess changed return and normal
cost jointly. If new/v98 tie while fixed-minus3 explains gains, mode accuracy is
not evidence for the current scalar feedback mechanism; the next design question
would be mode-conditioned control rather than another classifier adjustment.

## Limitations

Known mild Hopper modes, one trained source and short continuations; not a
cross-environment or independently trained-seed result. Empirical class weights
are not physical Bayesian posteriors. Frozen reference budgets differ across the
two readouts. Final evaluation is a small secondary pilot. Main-paper results
remain unchanged pending analysis.
