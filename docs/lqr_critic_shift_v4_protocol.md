# LQR critic_shift_v4: frozen factorial ablation

Frozen on 2026-10-01 before formal dispatch. This study isolates the positive critic weight-norm target shift identified in action_anchor_v3. The existing `weight_reg` coefficient is either 0.01 or 0; no loss implementation, global default, reward normalization, risk penalty or actor architecture is changed.

| Method | Action Anchor | Positive Target-Shift Coefficient |
| --- | ---: | ---: |
| csbapr | 0 | 0.01 |
| csbapr-action-anchor | 1 | 0.01 |
| csbapr-no-target-shift | 0 | 0 |
| csbapr-action-anchor-no-target-shift | 1 | 0 |

- All four arms train afresh with seeds 0/1/2, 500 episodes each: twelve formal runs. No v3 checkpoint initialization, seed expansion or coefficient search.
- Environment and shared settings remain those of action_anchor_v3: fixed 2D double integrator, amplitude-1 resets, 200 steps/episode, normalized actions [-1,1], physical force 5 times action, 1,000 random warmup transitions then one update per transition, batch 256, width 64, three critics, learning rates 0.0003.
- All arms use controlled degree-1 SINDy identification on separate amplitude-1 trajectories and the same fitted feedback, symbolic/Jacobian weights and arithmetic actor. Reseed training after identification. The replay deterministic-action anchor is unchanged except for its frozen coefficient 0/1.
- Final episode-500 checkpoint only; full-state episode-boundary resume every 20 episodes. Keep one atomic last checkpoint per run on its worker. No best/OOD-based selection.
- Evaluate amplitudes 1/2/5/10/20/50 with 20 paired seeds 10000--10019. Retain all returns, failures, origin action and initial-state action/Jacobian mismatch. Expected full budgets are 100,000 environment steps and 99,001 updates.
- Primary comparisons: removing the target shift within each anchor setting at every amplitude. Report the interaction (removal effect with anchor minus removal effect without anchor), ID regressions, and training-seed variability. Three training seeds do not establish significance. Historical BAPR/RE-SAC results are context only, not contemporaneous v4 baselines.
- Scheduler chooses idle compatible resources without node/GPU preferences. Stage an isolated frozen runtime and existing offline wheels. Synchronize only evaluation JSON; checkpoints stay on workers. A disjoint-seed CUDA startup/evaluation preflight is operational validation, not a scientific run.

## Limitations

This is a task-specific teacher under fixed linear dynamics and initial-state extrapolation, not general nonstationary RL or bus-style randomness. Removing the positive shift is an ablation, not a validation of the remaining BAPR design. Historical negative results and manuscript tables remain unchanged.

## Dispatch

The 22-test configuration/target/resume/diagnostic suite passes. The disjoint-seed-994 six-episode CUDA preflight finishes with exit code 0. All twelve formal tasks reached training with optimizer updates on 2026-10-01 across scheduler-selected local, jtl311linux, jtl110gpu, jtl110gpu2 and node007 resources; node007 GPU2/GPU3 are included. IDs are recorded in `experiments/lqr_critic_shift_v4_20261001/dispatch_manifest.json`. No manuscript result was promoted.
