# v142 budget-only continuation

Formal tasks t147938--t147943 submitted after three real full-state CUDA smokes.
Placement/status: `experiments/general_ood_v142_budget_extension_20261013/dispatch_manifest.json`.

Continue all six v141 final300k full states to600k, preserving each model's
actor/critic/target, optimizers, alpha, replay, belief/context, reward EMA, RNG
and environment state. No fresh initialization or checkpoint selection.
1.8M new interactions/updates; cumulative3.6M interactions /3598470 updates.

Actor lr3e-5, critic/alpha lr3e-4, MSE, gamma.99, tau.005, beta=-2, physics,
native rewards, bootstrap masks and evaluation remain unchanged. Preserve all
prefix blocks/evaluations; add the periodic evaluation skipped at the old300k
endpoint. At600k use the same CPU reset seeds154000--154004 and action noise
137000--137002,20 episodes/model. Record new and reused budgets separately.

Question: does stable ID seed3 need more training, and do the OOD short-survival
plateaus improve with budget alone? Retain all six outcomes. No rate sweep,
new seeds, environment relaxation or NAU/SINDy change. If ID recovers but OOD
plateaus persist, stop extending this rate/budget route and revisit physical
profiles before testing nonstationary adaptation.

Native CUDA smokes append512 updates to real300k states in separate directories;
source files are never overwritten. All three Linux nodes eligible with no
preference/pin; no Windows. Stage full states server-to-server only. Publish
small scalar JSON/log tails locally; checkpoint/replay remain on servers.

## Limitations
The primary comparison is each continued v141 trajectory versus its own300k
endpoint, not an independent replication. Original v137/v138300k results are
unmatched-budget references, not600k superiority controls.
