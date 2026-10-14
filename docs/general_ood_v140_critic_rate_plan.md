# v140 critic-rate isolation

Six formal tasks t147080--t147085 submitted after three native CUDA smokes.
Live placement and status: `experiments/general_ood_v140_critic_rate_20261012/dispatch_manifest.json`.

Fresh300k MSE policies, seeds3/4, profiles friction-only/gravity-only/ID: six
training tasks,1.8M new interactions. Reuse the corresponding v138 component
and v137 ID runs as the original-rate controls; preserve all their results.

Change only critic Adam learning rate3e-4 ->1e-4. Actor/alpha rates, gamma.99,
tau.005, reward EMA, context, beta=-2, physics, native rewards, training budget,
reset/noise seeds and final-not-best evaluation remain unchanged. Do not cap
alpha, change density formula, switch to Huber or soften physics in this round.

The v139 probes show huge behavior-action values and bootstrap terms, with
much smaller density-counterfactual target changes. A smaller critic step tests
whether reducing value/target feedback prevents divergence; it is not a causal
root claim. Inspect existing5k diagnostic blocks for Q/TD/alpha chronology,
then the exact v138 sampled-survival protocol at300k. Keep all runs, including
failed repairs; low TD error alone is not a locomotion success criterion.

Same-profile other-seed runs check whether the repair worsens already stable
training. ID seeds check normal locomotion regression. Stable OOD runs were
already short-lived: removing numerical explosion need not solve that separate
control problem. No new baseline-superiority claim or additional seed sweep.

All three native Linux nodes eligible; no preference or pin, no Windows.
Checkpoint/replay stay on servers; only scalar source JSON/log tails locally.
