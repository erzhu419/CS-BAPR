# v131 results and next step

All six tasks t143241--t143246 completed and synced645944 bytes of scalar JSON:
six sources /36 replay-noise cells /zero training or simulator steps. Five ran
on node007 CPU, one on jtl110gpu2 CPU; no preferences, Windows nodes, CUDA
initialization or local checkpoint downloads. Native smoke and16 tests passed.
Summary: `results/general_ood_v131_source_diagnostics/summary.json`.
Below: noise-averaged squared TD errors and pre-clipping critic gradient norms.

| Source | Uniform MSE | Last10k MSE | Top1% uniform loss share | Uniform critic gradient norm |
|---|---:|---:|---:|---:|
| Seed3 100k | 1.28e5 | 2.05e3 | 99.05% | 6.18e5 |
| Seed4 100k | 0.503 | 0.149 | 60.05% | 57.93 |
| Seed1 300k | 8.86 | 0.685 | 91.09% | 188.19 |
| Seed2 300k | 110.39 | 105.05 | 97.74% | 679.77 |
| Seed3 300k | 4.46e10 | 4.88e8 | 99.96% | 2.34e10 |
| Seed4 300k | 0.374 | 0.193 | 70.23% | 36.93 |

Seed3's100k loss is dominated by extreme replay/target tails; by300k recent
replay is unstable too. Its300k uniform Q mean is42253.24, median-7045.00 and
max1.41e7: the mean does not describe uniform positive inflation. Bootstrap
mean49401.76 dwarfs the entropy contribution-91.45. Minimum-Q targets and
gradient clipping at1 were already enabled. Seed4 remains numerically stable
but fails to acquire reliable locomotion. Tail concentration alone is not failure.

## Time-Limit Mask Error

The frozen v121 driver stores terminated-or-truncated at line115, incorrectly
zeroing bootstrap on external time limits; see
[Gymnasium's rule](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
Seed2's last10k has zero physical endings and10 time limits. Its four sampled
done transitions are therefore time limits and contribute98.1% of current TD
MSE: predicted Q327--333 versus reward-only targets4.10--4.76. Seed3's dominant
errors are non-done transitions, so this does not explain its early instability.

## Next Step

Plan v132 with fresh controls and terminated-only Bellman masks; episode resets
still use terminated-or-truncated. Compare MSE against2xHuber with fixed delta10
(same quadratic loss below10, bounded residual gradients above10), keeping all
other settings and the disagreement penalty. Seeds3/4, two arms,100k each:
400000 interactions /398980 updates. No sweep/filtering; final-only models and
the existing Linux pool. Planned, not dispatched; acquisition is a separate gate.

## Limitations

Frozen measurements establish neither training causality nor OOD gains. Replay
distributions differ; seed2's timeout attribution uses its original block counts.
Historical masks/actors/results and paper tables remain unchanged, not relabeled
as evidence under the proposed corrected protocol.
