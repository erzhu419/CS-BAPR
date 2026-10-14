# v138 results and next step

All six tasks t144792--t144797 completed exit0. Each trained300k steps /
299745 updates; total1.8M steps /1798470 updates. Matching component physics,
configuration, budgets and v137 control evaluation protocols verified.120 new
CPU episodes /12795 steps.665093 bytes of formal scalar JSON pulled; snapshots
remain on servers. Training took97.9--122.9 minutes per model.

| Condition | Seed3 sampled length /15 | Seed4 sampled length /15 |
|---|---:|---:|
| v137 ID control | 1000.00 | 1000.00 |
| Mass-only | 112.27 | 138.93 |
| Gravity-only | 91.87 | 69.07 |
| Friction-only | 74.13 | 148.07 |
| v137 compound-mega control | 77.20 | 73.20 |

All120 new episodes physically terminate. No component passes stable control
under this300k setup. Two numerical failures must be distinguished from the
four ordinary short-survival outcomes; retain their results, not a filtered mean.

Last5k block means: friction seed3 alpha=4.99e5, policy Q=4.40e8,
TD MSE=5.68e15; gravity seed4 alpha=1.36e6, policy Q=6.58e8,
TD MSE=3.18e17. Friction seed3 TD MSE rises from2.74 at15k to28.99 at30k
while alpha remains0.06--0.10. Gravity seed4 TD MSE reaches1303 at50k while
alpha is0.068. Q/TD growth thus precedes extreme temperature growth; the logs
do not establish either the causal root or an environment-control limitation.

Next v139, planned not dispatched: zero-training frozen-checkpoint/replay
diagnostics. Probe the two divergent models, their same-profile other-seed
controls and v137 ID seeds3/4: six CPU tasks, no optimizer updates. Read full
states only on servers. Stratify replay sampling by collection age; compare
behavior-action and policy-action Q, bootstrap/entropy/reward target terms,
actor saturation and native versus stable tanh log-density/gradients. Publish
small scalar JSON only. Diagnose stability before another environment sweep.

## Limitations
Two training seeds and this trainer/budget only. Numerical failures are not
physical-impossibility evidence; stable training with short survival is not
proof of irreducible environment difficulty. No adaptation or baseline
superiority established. Paper tables unchanged. Summary:
`results/general_ood_v138_physics_components/summary.json`.
