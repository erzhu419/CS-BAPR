# LQR feedback_v2: frozen development study

Frozen before dispatch on 2026-09-30. This tests the repaired symbolic interface
and smooth actor, independently of historical manuscript results.

- Methods: `csbapr`, `csbapr-no-sindy`, `csbapr-no-nau`, `bapr`, `resac`.
- Training seeds: 0, 1, 2; 500 episodes per run, at most 200 steps per episode.
- Training initial states: uniform four-dimensional unit ball; fixed dynamics
  and normalized actions in [-1, 1], with physical force equal to 5 times action.
- Shared budget: 1,000 random-action warmup transitions, then one gradient
  update per transition; batch 256, hidden width 64, three critics, actor and
  critic learning rates 0.0003. Exact settings are stored in evaluation JSON.
- Symbolic arms: degree-1 controlled SINDy, threshold 0.001, using only
  amplitude-1 identification trajectories. Identification has its own environment
  and does not advance training initial-state or exploration random streams.
- Action reference: tanh-squashed feedback from fitted dynamics, known Q and
  normalized-action cost R * 25; symbolic and full-Jacobian weights both 0.01.
- BAPR and RE-SAC retain the project's shared stabilization defaults. RE-SAC
  uses the existing fixed ensemble-risk preset without adaptive BAPR penalty.
  Neither baseline is a pristine reproduction of an external implementation.
- Selection: final episode-500 model only, with no OOD-based checkpoint choice.
- Evaluation: deterministic actions; amplitudes 1, 2, 5, 10, 20, 50; 20 episodes
  per amplitude; evaluation seeds 10000 through 10019 paired across every method
  and training seed. Retain individual returns and training curves, including failures.
- Primary comparison: CS-BAPR minus each comparator's mean return, separately
  at each amplitude; report all three training seeds. No post-result retuning.
- Checkpoint: one atomic full-state `last.pt` per run, every 20 episodes,
  including replay, all optimizers, RNGs, belief/surprise/reward statistics and
  episode counter. Resume starts at the next saved episode.
- Scheduler: no preferred or required node/GPU. Source is staged as a frozen,
  small runtime copy; each run has separate checkpoint and result directories.
  Checkpoints remain on workers. Only small evaluation JSON files are synced.
- Dependencies: a study-local virtual environment inherits an existing CUDA
  PyTorch >=2.4 installation; numerical/SINDy dependencies are pinned in
  `scripts/lqr_feedback_v2_requirements.txt`. Each result records actual library
  versions and GPU model. Existing project environments are not modified.

## Limitations
This study measures initial-state extrapolation under fixed linear dynamics,
not stochastic bus demand or general nonstationary dynamics. A positive result
does not establish broad RL applicability; a negative result is retained as such.
