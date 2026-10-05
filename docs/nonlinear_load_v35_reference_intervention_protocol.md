# v35: cost-linear learner intervention

Frozen 2026-10-04 (Asia/Shanghai), before numerical preflight or learner outcomes.
The independently passed v34 reference gate admits this one reference-only trial.

Use the unchanged sustained-load environment, seed21/22, ID-only identification
and training, reset radii 0.2/0.5/1.0, and 500 episodes of 200 steps per arm.
Compare cubic-local tanh and confirmed cost-linear clip references; both arms
retain Jacobian weight 0.01, full six-column Jacobians, reset-balanced Jacobian
sampling, early-critic replay, uniform actor replay, and entropy floor 0.01.
NAU, context, symbolic value loss, action anchoring and target shift stay disabled.
Use the exact confirmed gains; reproduce each ID physical fit within the existing
fit tolerances before training. No tuning, new learner baselines or extra seeds.

Share one complete five-episode prefix per seed, with Jacobian loss off and
uniform critic replay, then fork the full learner, optimizer, replay, temperature,
belief, environment and random/sampler states. Enable Jacobian loss and early
critic in both arms, changing only the reference. Fresh paired training is needed
for this full-state fork; v31's ablation is not pooled into this reference trial.
Actual training: 398,000 transitions / 396,002 updates across both seeds;
shared ID/validation: 22,000 transitions. Last checkpoint only; overwrite every
20 episodes and at completion. Recovery restores each arm's exact protocol.

Fresh formal paths 66000--66099, severity 1/50 at each radius, with actor, own
fitted, known-local and known-inverse controls: 9,600 episodes / 1,920,000
transitions. Evaluation calls the actual reference module, including its action
mapping and autograd Jacobian, rather than reconstructing tanh from its gain.
Numerical preflight uses seed21, six episodes per arm and separate paths
66500--66501; it checks fit/gain identity, full restore and sampler/cost accounting,
not performance. Unit tests cover saturated/interior derivatives and next-update
restoration. Clip has no finite smooth derivative certificate at saturation.

Retain individual-seed learner gates: the cost-linear learner must reduce total
and early state costs at radius 1 under both severities, not increase total cost
at radius 0.2, and beat its own fitted reference at every radius/severity/seed.
Only all three passing admits learner replication; failures stop without tuning
or threshold changes. Historical failed gates and manuscript HOLD remain intact.

Any free compatible Linux GPU, no preference, pin, allowlist or Windows. Keep
checkpoints/replay/trajectories remote; collect only logs and compact evaluation
JSON. Exclude `.runtime/` in every staging root and reuse pinned dependencies.
