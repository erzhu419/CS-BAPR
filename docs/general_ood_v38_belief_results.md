# v38 belief diagnostic results

Completed 2026-10-06: t135812--t135815 all done, exit 0, results synced. All 12 cells satisfy the frozen configuration, paired runtime/mode schedule, exact interaction budget, restored context count and evaluation protocol. Only 231681 bytes of formal JSON were needed; checkpoints remain remote.

## Results

Returns below average the two training-seed means; larger is better. The OOD vector preserves seed0/seed1. Reproduce with `python3 scripts/summarize_general_ood_v38.py`; all profiles, lengths, paired differences and descriptive seed-level intervals are in `results/general_ood_v38_belief/summary.json`.

| Environment | Variant | ID mean | Markov 50x mean | OOD seed0 / seed1 |
| --- | --- | ---: | ---: | --- |
| Pendulum-v1 | Legacy belief | -1208.58 | -1279.38 | -1213.30 / -1345.45 |
| Pendulum-v1 | Corrected belief | -1005.90 | -1190.97 | -1201.52 / -1180.43 |
| Pendulum-v1 | Fixed penalty | -1147.72 | -1261.94 | -1238.48 / -1285.41 |
| Hopper-v4 | Legacy belief | 1031.76 | 903.67 | 788.13 / 1019.22 |
| Hopper-v4 | Corrected belief | 534.80 | 319.38 | 578.54 / 60.22 |
| Hopper-v4 | Fixed penalty | 989.24 | 989.49 | 997.68 / 981.29 |

Pendulum corrected wins ID and Markov OOD against both controls on both seeds. Its OOD gaps are +11.78/+165.02 versus legacy and +36.96/+104.98 versus fixed. However, its worst fixed profile is recovery_aftershock, worse on both seeds: -47.00/-16.02 versus legacy and -27.91/-21.61 versus fixed. This fails the frozen floor condition, even though average OOD improves.

Hopper corrected loses Markov OOD against both controls on both seeds. Seed1 also collapses on ID: 60.22 return and 36.8 steps versus approximately 1000 return/1000 steps for controls. Its Markov profile counts contain zero non-ID transitions: all episodes finish before the first switch at step200. This is training/ID failure, not demonstrated failure to adapt after an OOD switch. Seed0 has a genuine exposed OOD regression, including heavy_traffic return416.32 versus legacy1044.52 and fixed997.54.

## Mechanism

Pendulum corrected belief responds and recovers: first-block surprise5.88/5.01 gives a window2.15/2.65; final-block surprise0.88/0.86 gives window14.43/14.32 (seed0/seed1). Its stronger penalty is transient, unlike the Hopper seed1 failure.

Hopper corrected seed1 is effectively saturated from the 10k--20k block onward: reset probability approximately1, window approximately0 and beta=-3. Final-block alpha43536, Q-std977546 and critic loss1.61e15 reveal runaway training, despite finite result JSON and a successful task exit. Matched controls end at alpha2.99/3.35 and Q-std48.57/52.83. Corrected seed0 instead recovers to reset probability0.044 and window4.91.

The source combines reward z-score, relative Q-std spike and an absolute critic/target L1-norm difference using max then EMA. The last term scales with network size, whereas the new filter assumes base variance1 in both environments. Hopper first-block surprise is roughly88--98 versus Pendulum5--6. This makes surprise calibration a concrete suspect; per-component values and reward-normalization scale were not recorded, so the triggering component and causal feedback path remain unresolved.

## Decision And Next Step

HOLD: neither environment passes the frozen admission rule. Preserve the Pendulum gain and all adverse seeds; do not expand confirmation seeds or update manuscript superiority claims from this round.

Next diagnostic should log surprise components, reward scale, entropy and critic regularization separately; make the critic-divergence input dimensionless rather than tuning a Hopper-only coefficient. Include a constant beta=-3 control to distinguish adaptive timing from a stronger constant penalty, starting with the failed Hopper seed1 under a short, equal-interaction diagnostic. No new tasks were submitted during this completion analysis.

## Limitations

Two training seeds and reduced Hopper budget are development evidence, not final convergence or general transfer confirmation. Fixed penalty is the matched RE-SAC-style arm, not the differently configured v37 RE-SAC. NAU/SINDy/Jacobian losses are absent; belief updates only during training, while evaluation context updates online. No pooling with v37, no significance claim from evaluation-episode counts, and no locomotion-quality claim from Hopper reward alone.
