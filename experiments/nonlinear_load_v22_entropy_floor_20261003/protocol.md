# Nonlinear load v22: entropy-floor-only intervention

v21 found lower radius1 cost after replacing only the first action. Entropy reversed the mean risk ranking in seed15/ID but seed16 already had a pre-entropy mismatch. Test the floor as a causal hypothesis, without assuming it explains both seeds.

## Frozen Experiment

Fresh matched seeds17/18; arms `floor` and `no-floor` differ only in `enable_entropy_floor` (true/false). Both retain automatic temperature learning and configured nominal floor0.01. The switch affects the shared temperature in the actor objective and critic entropy backup, not only the actor term.

Retain v20 full six-input Jacobian weight0.01, centered tanh MLP, cubic-local fitted reference, unchanged auxiliary linear model, 50% stored-reset/50% replay Jacobian sampler and independent sampler seed100000+training_seed. Environment, reward EMA, risk/belief updates, optimizer rates, reset cycle0.2/0.5/1, severity1 training and identification remain unchanged. Per arm:10,000 identification plus1,000 validation transitions,500 training episodes/100,000 interactions/99,001 updates, save every20 episodes and evaluate last only. Four runs total400,000 learner interactions/396,004 updates.

Evaluate radii0.2/0.5/1, loads1/50 and paired paths60000--60099. Each run evaluates actor/own fitted/known local/known inverse feedback:2,400 controller episodes,9,600 total, without learning. Record per-episode alpha, unconstrained alpha, reward standard deviation, and cumulative updates below the nominal floor/with binding floor; restore these records on resume. Keep each seed/load separate.

Admission gates are unchanged: the no-floor arm must strictly improve radius1 total and early-state cost on each seed/load, retain radius0.2 total cost with zero margin, and strictly beat its own fitted reference in every radius/seed/load block. All gates must pass before learner-comparator replication; otherwise HOLD. No threshold, checkpoint, seed, environment or weight retuning. Historical HOLD and manuscript tables remain unchanged.

## Preflight And Placement

Tests verify the single config difference, exact native floor-arm updates, inactive-floor arm equality, effective below-floor intervention, resume/temperature history and individual gate failures. Two seed17 CUDA preflights run six episodes, save/reload after episode5, and evaluate two paired paths per block. Require201 updates,25,728 reset plus25,728 ordinary Jacobian samples, matched identification/first-five exploration histories/direct-controller returns, finite values, conserved costs and correct temperature records. At201 updates the floor may remain inactive; this is runtime admission, not treatment-effect evidence.

Scheduler selects idle compatible Linux GPUs without preference, pin or allowlist; no Windows. Checkpoints/replay stay on servers outside result directories. Return only small JSON/log artifacts; no runtime optimization. Frozen study: `experiments/nonlinear_load_v22_entropy_floor_20261003`; results: `results/nonlinear_load_v22_entropy_floor`. Summarize completed formal tasks with `python3 scripts/summarize_nonlinear_load_v22.py` in the frozen package runtime.

## Limitations

Two seeds in one fully observed environment do not establish general RL or BAPR/RE-SAC superiority. v21 pointwise scores are not expected policy objectives or critic-error certificates. This ablation tests the entropy-floor switch; it does not isolate entropy from its effect on learned critic targets, nor certify reference optimality.
