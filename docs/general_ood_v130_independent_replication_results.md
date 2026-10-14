# v130 results and next step

All 13 formal tasks completed successfully. The frozen merger validated five
training seeds, 15 final actors, 1575 episodes and 576889 evaluation interactions,
with zero evaluation updates. New training cost: 800000 interactions / 799490
optimizer updates. Both added Linux nodes passed native setup and smoke checks;
the four continuation arms ran on jtl110gpu/jtl110gpu2 (RTX 3080 Ti), while source
training ran on node007 (RTX 2080 Ti). Only compact task JSON returned locally
(3246898 bytes); no checkpoint/replay download. Summary:
`results/general_ood_v130_independent_replication/summary.json`.

Sampled evaluation: scale one, resets151000--151004, noise14000--14019,
100 episodes per model/seed, native 1000-step Hopper-v4 limit. The following
comparison is matched at 350k training interactions per arm.

| Training seed | Control return | Retention return | Control falls / 100 | Retention falls / 100 |
|---|---:|---:|---:|---:|
| 0 (prior) | 429.96 | 420.09 | 100 | 100 |
| 1 (prior) | 2372.66 | 3243.26 | 32 | 7 |
| 2 (prior) | 268.24 | 3015.74 | 100 | 9 |
| 3 (new) | 10.14 | 11.92 | 100 | 100 |
| 4 (new) | 438.64 | 440.54 | 100 | 100 |

The large gains on prior seeds1/2 persist, but the independent seeds3/4 do not
replicate comparable survival gains. Their mean retention lengths are 11.89 and
147.18 steps. Deterministic evaluation also falls in all five episodes for
seeds0/3/4; both 350k arms finish all five for seeds1/2. Seed2's 300k reference
finishes all 100 sampled episodes, versus nine retention falls: retention only
partially prevents deterioration. The 300k reference is not a matched-budget arm.

## Source Diagnosis

Seed3 fails before mean retention is introduced. Source diagnostic blocks show
critic loss 17.84 at50k, 69428.80 at100k, 5.54e7 at150k and 2.30e10 at300k.
Mean Q rises from150.83 at100k to52037.39 at300k; alpha rises from0.184 to32.841.
Periodic diagnostic returns fall from457.12 at100k to17.49 at150k. On the same
five final-evaluation resets150000--150004, source length drops from150.6 at100k
to11.0 at300k. Reward-normalization std stays near1.1, not near zero. The final
512-transition target probe contains discounted minimum-next-Q 52705.65,
entropy contribution -99.58 and zero L1 uplift; bootstrap Q dominates this probe.
Minimum-Q targets and gradient clipping were already enabled. These observations
locate source instability, but do not identify its causal mechanism.

Seed4 is a separate acquisition failure: Q stays near150, final-block critic
loss is0.51 and alpha0.0146, yet source length remains about145 steps. Stabilizing
seed3 alone cannot establish reliable source learning across seeds.

## Next Step

Plan v131 as a frozen source diagnostic, not another retention-coefficient or
seed sweep. Use existing server-side 100k sources3/4 and 300k sources1/2/3/4.
On predeclared uniform and recent replay slices, report per-critic TD-error
quantiles/tail loss share, terminal versus nonterminal residuals, bootstrap and
entropy terms, action/log-probability scales and actor/critic gradient norms.
No optimizer updates, environment interactions or checkpoint downloads. This
distinguishes concentrated replay/target outliers from broad Q inflation before
choosing one controlled training intervention. It is planned, not dispatched.
Use any idle eligible Linux node, without preferences or Windows nodes.

## Limitations

Only seeds3/4 are new independent replication; reset/noise repeats are not
training seeds. The intervention currently supports preservation of useful
sources, not dependable acquisition, OOD superiority or submission readiness.
Sources and continuations used different GPU models. Earlier negative results
and paper tables remain unchanged; no weak seed or periodic model was selected.
