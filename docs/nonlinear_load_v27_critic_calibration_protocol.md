# Nonlinear load v27: ID-only critic direction calibration

Freeze the two v25 early actors (seeds19/20), their fitted references, alpha, reward standard deviation and risk coefficients exported for v26. Compare each original critic with a calibrated clone. Keep v25's failed own-fitted performance gate and historical HOLD unchanged.

Calibration only: load1, radii0.2/0.5/1, disturbance paths61000--61015 and four Gaussian-noise seeds93000--93003. These do not overlap v26's evaluation paths60000--60099 or noise92000--92007. Each radius reuses the16 paths, giving48 initial states per model. Perturb only the initial latent mean by +/-0.005 along its unit fitted-reference direction; preserve initial standard deviation, then follow the frozen stochastic actor for200 steps with common noise/disturbances. Labels are the noise-averaged central difference of normalized discounted reward plus future entropy (steps1--199), with no terminal bootstrap or initial entropy. Either label sign is valid.

Fit the clone using full-batch Adam at lr0.0003 for exactly1,000 updates, gradient clip1, and final-update selection. Loss = mean squared error between the critic ensemble-mean paired direction and its Monte Carlo label, plus weight1 mean squared displacement of individual original Q values at the same paired actions. The value anchor limits unrelated value/ensemble drift; it does not certify unchanged uncertainty. No hyperparameter sweep or test-set checkpoint selection.

Evaluate original/calibrated Q, risk and soft actor scores on v26's fixed initial states, with its full100-state Gaussian batches, both step sizes and eight draws. Reuse actual v26 continuations and per-draw return means: the actor never changes, so these outcomes remain valid without rerunning64,000 trajectories. Reproduce original scores before interpretation. 50x states/returns never enter fitting.

Formal label budget across two tasks:768 controller episodes,153,600 simulator transitions,2,000 retained optimizer updates; evaluation adds zero simulator transitions. Seed19 GPU preflight uses8 episodes/1,600 transitions and5 updates; its score comparison still uses the full v26 evaluation grid. Save critic/Adam/data/progress checkpoints remotely every100 updates and at the final update. Sync only compact JSON; raw labels and checkpoints remain remote. Dispatch one paired comparison per seed to whichever compatible Linux GPU is free, without preferences/pins/allowlist or Windows nodes.

## Interpretation

This is an offline critic-fit feasibility experiment, not a policy-performance improvement. Compare ID and50x local ordering, actor-score residuals and small-radius behavior; retain negative transfer. Local reference directions and reused evaluation paths do not provide a full actor-gradient certificate or fresh performance confirmation. Numerical/identity preflight admits execution only. Choose an online intervention only after the two fixed-budget results.
