# RandomLoad validation v8: completed results

Eight tasks `t118773`--`t118780` finished with exit code 0. All final JSONs match the frozen configurations, 500 episodes, 100,000 learner interactions, 99,001 updates and evaluation seeds 20000--20019. All used Python 3.11.15 and Torch 2.5.1+cu121. Final JSONs were already local; this completion pass transferred no remote data or checkpoints. Cost is minus return; lower is better. Means/SDs are over the two independent training seeds 3/4.

| Method | 1x | 2x | 5x | 10x | 20x | 50x | 50x Seed SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centered/no-NAU + symbolic | 0.149067 | 0.149416 | 0.152807 | 0.166704 | 0.236451 | 0.969016 | 0.001077 |
| Centered/no-NAU, no symbolic | 0.268542 | 0.271324 | 0.284240 | 0.358436 | 0.619954 | 3.789975 | 3.435956 |
| BAPR | 22.476080 | 22.309844 | 22.508774 | 24.949514 | 39.359732 | 353.028891 | 311.779718 |
| RE-SAC | 68.509552 | 68.529564 | 67.361341 | 70.550473 | 97.206597 | 581.590802 | 354.159823 |

## Decision

Independent replication passes against all three learners: lower mean ID/50x cost and 2/2 matched-seed wins at both, with 2/2 wins at every other severity too. The selected candidate's 50x seed costs are 0.968255 and 0.969777. Its mean 50x cost is 74.43% below no-symbolic, 99.73% below BAPR and 99.83% below RE-SAC. Retain the centered/no-NAU symbolic candidate; the v7 symbolic-removal gates remain failed.

Direct fitted feedback costs 0.148879 at ID and 0.967274 at 50x, versus learner costs 0.149067 and 0.969016. Learner-minus-fitted-feedback returns at 50x are -0.000933 and -0.002551; ID gaps are also negative on both seeds. The predeclared added-RL-value gate therefore fails (0/2 wins at both ID and 50x). The learner closely recovers the fitted controller, costing 0.18% more at 50x; an advantage over that controller is not established. Known-model feedback costs 0.967393 at 50x and remains a separate privileged reference. Gain matrices and raw paired episode gaps are saved.

## Limitations

Fixed linear dynamics, fully observed load and a known equilibrium limit the scope; this is not general RL or bus-level transfer. Two validation seeds do not establish significance. Symbolic arms additionally use 11,000 identification/validation interactions; the bundle includes both critic and actor constraints. GPU models differ (RTX 2080/3080 Ti), although recorded numerical package versions match. V7 and v8 evaluation paths differ, so their absolute costs are not a before/after improvement metric and their seeds are not pooled. Direct tanh feedback is feasible, not asserted optimal. Reset-state alignment does not describe complete policy trajectories. Prior negatives and manuscript tables remain unchanged.

## Next Step

Keep this candidate fixed and stop expanding the linear task. Define a controllable nonlinear ongoing-load environment, then freeze it using non-learning control pilots before observing any learner ranking. Preserve extreme stochastic load, observable current disturbance and bounded control authority; retain direct fitted/known-feedback controls. Freeze the environment and comparisons before formal training. No new training was dispatched by this completion pass.

Reproduce: `experiments/random_load_validation_v8_20261001/runtime/.runtime/bin/python scripts/summarize_random_load_validation_v8.py`. Evidence: `results/random_load_validation_v8/summary.json`, `seed_mean_returns.csv`, `task_observations.json`, `completion.json` and eight final evaluation JSONs.
