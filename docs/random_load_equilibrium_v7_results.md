# RandomLoad equilibrium v7: completed results

All 12 effective tasks finished with exit code 0. The frozen configurations, 500 episodes, 100,000 learner interactions, 99,001 updates and 20 paired evaluation seeds (10000--10019) per severity match all final JSONs. Failed startup attempts remain in the dispatch manifest and contribute no trained models. This completion pass recovered two small local-worker JSONs; it transferred no remote data or checkpoints. Manuscript tables are unchanged.

Cost is minus return; lower is better. Means and sample SDs are over three training-seed means. Every new arm has no NAU, target shift or action anchor.

| New Arm | 1x | 2x | 5x | 10x | 20x | 50x | 50x Seed SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain + symbolic | 8.924742 | 8.927084 | 8.932131 | 8.953884 | 9.073888 | 10.126898 | 12.964521 |
| Plain, no symbolic | 16.266702 | 16.259179 | 16.306086 | 16.674873 | 17.919972 | 27.150884 | 9.214981 |
| Centered + symbolic | 0.140601 | 0.140747 | 0.144584 | 0.170427 | 0.275513 | 1.199133 | 0.001521 |
| Centered, no symbolic | 1.955074 | 1.959048 | 1.973633 | 2.072564 | 2.448268 | 6.517403 | 1.802023 |

## Decision

Freeze `csbapr-centered-mlp-no-target-shift` as the accuracy-first candidate: drop NAU, retain the symbolic bundle and train with `tanh(mu(s)-mu(0))`. Both centering comparisons pass the predeclared lower-ID/lower-50x-mean plus 3/3 matched-seed-win gate. Removing the symbolic bundle fails the gate in both architectures, with 0/3 wins at ID and 50x. All four contrast directions hold at every tested severity; do not change the removal gate.

Centering lowers 50x mean cost by 88.16% with the bundle and 76.00% without it. Centered + symbolic costs 81.60% less than centered/no-symbolic; its three 50x seed costs are 1.199348, 1.197516 and 1.200536. The lightweight no-NAU/no-SINDy variant remains a useful alternative, not an accuracy-equivalent replacement. The paired return-scale centering-by-symbolic interaction at 50x is -11.705717 (seed SD 5.108142): absolute centering gains are smaller with the already-better symbolic arm, so this is not positive additive synergy.

Both centered actors map zero exactly to zero. At 50x reset observations, mean action RMSE versus each arm's reference is 0.0116 with the bundle and 0.1513 without; squared-Frobenius Jacobian error is 0.0479 versus 4.8601. These diagnostics support improved action slopes beyond removing the offset.

Historical v6 BAPR/RE-SAC 50x costs are 907.568317/529.583966. The identical frozen environment and paired evaluation paths permit labeled historical comparisons. Known-model feedback plus force compensation costs 1.196847 at 50x; the best learner costs only 0.19% more than this feasible reference.

## Limitations

This task has fixed linear dynamics, fully observed load and a known zero equilibrium. It establishes a successful ongoing-load stress test, not general RL or bus-level transfer. The near-reference result leaves the learner's added value over direct fitted feedback unresolved; the known-model tanh feedback is not an optimum. The symbolic bundle removes/adds both critic target penalty and actor Jacobian alignment, together with 11,000 identification/validation interactions (111,000 versus 100,000 total), not an isolated, data-matched SINDy effect. Alignment metrics cover reset observations, not full policy trajectories, and use fitted versus known references. Three training seeds do not establish significance. Plain seed 0 uses Torch 2.8/RTX 4060 while centered seed 0 uses Torch 2.5/RTX 2080 Ti; other centering pairs and all symbolic-removal pairs match recorded runtimes. Baselines are non-contemporaneous. The repeated plain-symbolic arm replaces, rather than augments, v6's same seeds.

## Next Step

Keep this environment, weights and budget frozen. Validate on independent training seeds 3/4 for centered + symbolic, centered/no-symbolic, BAPR and RE-SAC (eight training tasks), with disjoint paired evaluation seeds 20000--20019 and direct fitted/known-feedback controls. Use one compatible software build and let the scheduler choose any idle compatible node/GPU. Retain v7 as the development result; report new seeds separately before combining evidence. After replication, test a separately frozen controllable nonlinear environment where fitted feedback is approximate. Select that environment by non-learning controllability checks, not by the learned method ranking. No new training was dispatched by this completion pass.

Reproduce: `experiments/random_load_equilibrium_v7_20261001/runtime/.runtime/bin/python scripts/summarize_random_load_equilibrium_v7.py`. Evidence: `results/random_load_equilibrium_v7/summary.json`, `seed_mean_returns.csv`, `task_observations.json`, `completion.json` and the 12 final evaluation JSONs.
