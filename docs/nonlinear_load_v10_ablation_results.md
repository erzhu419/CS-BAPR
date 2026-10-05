# Nonlinear load v10: completed symbolic-loss ablation

All four tasks completed with exit 0, taking 31.5--78.1 scheduler wall-clock minutes. Frozen configurations, 100,000 learner interactions, 99,001 updates, pinned packages and all 100 paired paths match. Fitted gains and direct-controller returns match same-seed v9 controls. The four JSONs were already local (584,429 bytes); this completion pass transferred no checkpoints or remote data. There are 9,600 new evaluation episodes including controls.

Mean cost across training seeds 5/6, lower is better:

| Arm | ID | 50x |
|---|---:|---:|
| Full symbolic (v9) | 0.258902 | 3.791397 |
| Critic-only | 5.240351 | 23.482876 |
| Jacobian-only | 0.258427 | 3.688081 |
Jacobian-only passes the frozen ID/50x comparison against no-symbolic/BAPR/RE-SAC and the separate added-RL-value criterion, all with 2/2 matched-seed wins. Its own-fitted return gaps are +0.002001/+0.002796 at ID and +0.111397/+0.071841 at 50x: a 2.42% mean 50x improvement. Both 50x seeds improve over full-symbolic, lowering mean cost by 2.73%; ID is not unanimously better (1/2 wins). Critic-only fails added value and comparisons against no-symbolic/BAPR, despite improving mean cost over no-symbolic. Actor Jacobian regularization is the dominant contribution here; retain the fitted reference, not the critic symbolic term. Historical v9's failed gate is unchanged.

## Next Step

Freeze centered/no-NAU, critic symbolic weight 0 and actor Jacobian weight 0.01. Confirm on independent seeds 7/8 against no-symbolic, BAPR and RE-SAC (eight tasks), retaining own-fitted controls and the same environment, budgets and gates. Do not rerun the rejected critic-only arm or retune severity. No new training was submitted by this completion pass.

## Limitations

Only two trained policies support this selection; 100 evaluation paths are not 100 training replications. Reset-action RMSE remains about 0.012 against the fitted reference, and Jacobian-only still costs 35.3% more than privileged inverse feedback at 50x. Seed5's worst fitted-feedback gap is -2.234678 on path 50002, worse than full-symbolic's worst gap despite the improved mean. This is not pathwise dominance, broad RL generalization or optimality. GPU/Python variants differ; pinned scientific packages match. Manuscript tables remain unchanged.

Reproduce with `python3 scripts/summarize_nonlinear_load_v10.py`. Full six-severity statistics, all paired gaps and failed gates are in `results/nonlinear_load_v10_ablation/summary.json`; all 24 new seed/severity means are in `seed_mean_returns.csv`.
