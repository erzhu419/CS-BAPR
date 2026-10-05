# v31: symbolic Jacobian on/off in sustained load

Frozen on 2026-10-04 (Asia/Shanghai), before numerical preflight or learner outcomes.

## Design

Use the v30-qualified `SustainedNonlinearLoad-v0`: always-active bounded
heavy-tail force, base cap 0.08, refresh probability 0.1, horizon 200, unchanged
plant, cubic actuator, observations and cost. Training and identification use
severity 1 only. Evaluation uses severity 1 and 50; no severity labels or future
loads are inputs.

Fresh training seeds 21/22. Each seed runs off then on sequentially on one
automatically selected Linux GPU. The sole per-arm configuration difference is
`jac_weight`: off = 0, on = 0.01. Both retain the same ID-fitted cubic-local
reference; off is a matched symbolic-loss ablation, not original BAPR. NAU,
context, target shift, symbolic value loss and action anchoring remain disabled.

Retain v25 early-critic sampling after the shared prefix: 50% native uniform,
25% stored resets, 25% steps 1--49. Actors remain uniform, rejecting v29 early
actor coverage. Jacobian inputs cover all six columns and use 50% native uniform
plus 50% stored resets. Off advances this same private sampler without computing
the loss. The same private sampler streams and counts must match across arms.

One complete five-episode prefix (1,000 transitions and one update) per seed,
with uniform critic and Jacobian loss off. Fork the full actor, critics, targets,
Adam states, temperature, belief, replay, reference, random generators and sampler
states; do not reset optimizers. Then enable early critic in both arms and the
Jacobian loss in on. Reset radii cycle 0.2/0.5/1.0 by absolute episode index.

Each arm ends at episode 500: 100,000 transitions and 99,001 updates. Actual
two-seed training: 398,000 transitions, 396,002 updates after shared-prefix reuse.
ID/validation: 10,000/1,000 transitions per seed, shared across arms (22,000 total).
Checkpoint selection is last only; overwrite each arm's full last every 20
episodes and at completion. Recovery uses each arm's own exact-protocol last;
the shared prefix remains available remotely.

Evaluation explicitly constructs the sustained environment. Use fresh paths
62000--62099 at each radius and severity, with actor, own fitted, known-local and
known-inverse controllers: 9,600 episodes / 1,920,000 transitions total. No
optimizer or identification updates occur during evaluation.

## Decisions

Retain the existing individual-seed gates: on must reduce both total cost and
early state cost at radius 1 under each severity; total cost at radius 0.2 must
not increase; on must beat its own fitted control at every radius/severity/seed.
Only all three passing admits learner replication. No pooling, threshold
relaxation, extra seeds or new learner baselines in this round.

Numerical preflight: seed21, six episodes per arm, two evaluation paths. It
checks finite outputs, environment identity, shared prefix, matched controls and
samplers, loss weights, runtime and interaction accounting. It is not a
performance gate and cannot tune this protocol.

## Limitations

v30 establishes control feasibility and sustained command cost, not sustained
state difficulty, a hidden OOD regime or a bus-like algorithm advantage. Existing
negative learner evidence and manuscript HOLD remain unchanged. All checkpoints,
replay and raw trajectories stay remote; pull only logs and compact evaluation
JSON. Scheduler chooses any free compatible Linux GPU, without preference, pin,
allowlist or Windows placement.
