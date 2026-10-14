# v141 slower actor, original critic

Formal tasks t147479--t147484 submitted after three native CUDA smokes passed.
Placement/status: `experiments/general_ood_v141_actor_rate_20261013/dispatch_manifest.json`.

Fresh300k MSE policies, seeds3/4, friction-only/gravity-only/ID: six native CUDA
tasks,1.8M interactions /1798470 updates. Primary paired controls are the original
v138 component and v137 ID runs. Preserve the v140 negative results as secondary
contrasts; v141 versus v140 changes two rates and is not a single-variable contrast.

Against the primary controls, change actor lr1e-4 ->3e-5 only; critic remains3e-4.
Keep alpha lr3e-4 with no cap, gamma.99, tau.005, reward EMA, context, beta=-2,
physical multipliers, native rewards, termination-only bootstrap, random seeds,
diagnostic/evaluation cadence and final-not-best endpoint unchanged. Same final
CPU reset seeds154000--154004 and noise seeds137000--137002,20 episodes/model.

Hypothesis: slower policy movement may limit chasing insufficiently calibrated
Q while retaining the original critic learning speed. Examine Q/TD/alpha
chronology, all same-profile survival outcomes and ID regression, not low TD
error alone. No new seeds, physics relaxation, NAU/SINDy or threshold sweep.

Core frozen from v140. Three profile-specific CUDA smokes must show only the
actor-rate config change and correct training/resume/evaluation behavior before
formal release. All three Linux nodes eligible, no preference or pin; no Windows.
Small source JSON/log tails locally; checkpoint/replay only on servers.

## Limitations
This tests a stability intervention, not adaptation or baseline superiority.
Only two training-seed pairs; numerical stability does not imply OOD controllability.
