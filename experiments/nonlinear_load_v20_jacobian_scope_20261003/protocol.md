# Nonlinear load v20: release physical-state derivative matching

v19 lowered all12 actor-cost blocks after repairing reference identification but failed radius1 early regulation and every own-reference added-value block. The next hypothesis is that constraining all observation-input derivatives restricts useful physical-state regulation. Freeze the environment and cubic-local reference; vary only the Jacobian input columns.

## Intervention And Budget

Fresh seeds15/16, two arms each: `full` retains all six input columns; `load-only` retains columns4/5 (observed external loads). In each arm, sum squared differences over retained action/input entries and average over sampled states. Keep weight0.01; no column-count normalization or rescaling. Full scope must reproduce native v19 updates at the same seed. Both arms retain the same50% ordinary-replay/50% stored-reset Jacobian sampler, independent seed100000+training_seed.

Use the unchanged centered tanh MLP without NAU, action anchoring, target shift or critic-symbolic loss. Both arms use the same per-seed10,000 identification and1,000 validation transitions, fit both original linear and cubic models at threshold0.001, select the cubic fit's frozen local feedback and retain the unused auxiliary linear world model. Training uses severity1,500 episodes/100,000 interactions/99,001 updates, balanced absolute-episode reset cycle0.2/0.5/1. Save every20 episodes and evaluate last only. Learner trajectories may diverge after the actor intervention.

## Evaluation And Frozen Gates

Evaluate radii0.2/0.5/1 and loads1/50 on100 paired paths60000--60099. Each run evaluates actor, own fitted reference, known local and known inverse feedback:2,400 controller episodes with zero learning/fitting. Retain early50/late150 state/command costs and initial/trajectory Jacobian alignment.

Load-only must strictly improve radius1 total and early state cost versus full for each seed/load, preserve radius0.2 total cost with zero margin, and strictly beat its own fitted reference at every radius for each seed/load. Only all gates passing admits learner-comparator replication. Otherwise HOLD, without retuning seeds, weights, columns, identification, reset ratios or checkpoints. Historical v13/v15/v17/v19 HOLD and manuscript tables remain unchanged.

## Verification And Dispatch

Focused tests must establish the retained-column loss/gradients, exact native full-scope updates, matched identification and initial critic/alpha/belief updates, effective actor intervention and exact full-state continuation for each scope. Two seed15 CUDA preflights run six episodes, saving/reloading after episode5 before the last200 steps. Require201 updates,25,728 reset plus25,728 ordinary Jacobian samples, identical first-five exploration histories and reference metadata, matched direct-controller returns and explicit scope restoration. Preflights are runtime admission, not efficacy evidence.

Scheduler selects idle compatible Linux GPUs without node/GPU preference, pin or allowlist; no Windows. Checkpoints remain on servers outside result directories. Return only small logs/JSONs. Do not optimize runtime.

## Limitations

This is one direction-selection intervention in a disclosed, fully observed environment; it is not a new general RL benchmark or an optimality certificate. Fewer retained columns can lower the total penalty even at fixed per-entry weight, so any improvement concerns the combined physical-column removal, not an equal-total-regularization comparison. Two fresh seeds and reused disturbance paths bound inference. Direct controls are feasible references, not certified optima.
