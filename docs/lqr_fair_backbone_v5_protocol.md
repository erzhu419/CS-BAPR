# LQR fair_backbone_v5: frozen fair-target comparison

Frozen on 2026-10-01 before formal dispatch. The v4 factorial identifies a harmful positive critic weight-norm bootstrap shift. This study asks whether CS-BAPR retains an advantage when that shift is also removed from its baselines and component ablations. No learner implementation, environment, training-efficiency setting or global default changes.

| Method | Actor | Fitted Symbolic Alignment | Target Shift | Action Anchor |
| --- | --- | --- | ---: | ---: |
| csbapr-no-target-shift | Smooth NAU | Yes | 0 | 0 |
| bapr-no-target-shift | ReLU MLP | No | 0 | 0 |
| resac-no-target-shift | ReLU MLP | No | 0 | 0 |
| csbapr-no-sindy-no-target-shift | Smooth NAU | No | 0 | 0 |
| csbapr-no-nau-no-target-shift | Tanh MLP | Yes | 0 | 0 |

- Five arms, fresh seeds 0/1/2, 500 episodes each: 15 formal runs. No warm-start, new seed, coefficient search or anchored arm. The repeated CS-BAPR arm is a contemporaneous control, not a fourth independent seed.
- Fixed 2D double-integrator environment, amplitude-1 training resets, 200 steps/episode, normalized actions [-1,1], physical force 5 times action. Shared width 64, three critics, batch 256, learning rates 0.0003, 1,000 random warmup transitions, then one update per transition. Expected budgets: 100,000 environment steps and 99,001 updates per run.
- Every arm differs from its established preset only in `weight_reg=0`. CS-BAPR and the no-NAU arm fit degree-1 SINDy on separate amplitude-1 trajectories, then reseed training. Their symbolic/Jacobian weights remain 0.01; actor weight decay remains 0.0001. The no-SINDy arm omits identification and both alignment losses. Arithmetic regularization remains 0.01 where applicable.
- BAPR and RE-SAC retain their shared inherited stabilization and behavior-cloning weight 0.001; CS-family presets retain behavior-cloning weight 0. RE-SAC retains fixed ensemble-risk control; BAPR retains adaptive belief risk. No hidden baseline retuning.
- Use only the final episode-500 checkpoint. Full-state episode-boundary resume every 20 episodes; one last checkpoint per run stays on its worker. Evaluate amplitudes 1/2/5/10/20/50 with paired seeds 10000--10019, 20 episodes per amplitude. Retain all returns and action/Jacobian diagnostics; no best/OOD selection.
- Primary comparisons: CS-BAPR versus BAPR and RE-SAC at each amplitude, with training-seed means and matched-seed differences. Secondary comparisons remove symbolic alignment or replace the arithmetic actor. Report ID regressions and every negative contrast. These are preset-level comparisons, not an estimate of each individual loss effect.
- Use scheduler-selected idle compatible resources without node/GPU preferences. Freeze source and stage existing offline wheels; synchronize only small evaluation outputs, never checkpoints. A disjoint-seed-995 six-episode CUDA preflight validates operation only.

## Limitations

Three training seeds do not establish statistical significance. Inherited baseline stabilization means these are this repository's implementations, not pristine reproductions. Mixed worker Torch/CUDA builds are retained and recorded. This remains task-specific feedback under fixed linear dynamics, not general RL or bus-style stochastic OOD. Historical results and manuscript tables remain unchanged.

## Dispatch

The 30-test preset/target/resume/diagnostic suite passes; the disjoint-seed-995 CUDA preflight exits 0 with 1,200 environment steps and 201 updates. All 15 formal runs reached training with positive optimizer-update counts on 2026-10-01, across scheduler-selected jtl311linux, jtl110gpu, jtl110gpu2 and node007 (including GPU2/GPU3). No node/GPU preference is set. Task IDs t116593--t116607 and per-run startup observations are recorded in `experiments/lqr_fair_backbone_v5_20261001/dispatch_manifest.json`.

All 15 runs subsequently completed the frozen budgets and final paired evaluations. Results and component-negative contrasts are recorded in `docs/lqr_fair_backbone_v5_results.md`; manuscript tables remain unchanged.
