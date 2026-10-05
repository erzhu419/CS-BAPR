# v32: actuator-aware reference diagnostic

Frozen 2026-10-04 (Asia/Shanghai), before any v32 rollout outcome. v31's symbolic
on/off comparison improves all 12 blocks, but on fails the own-fitted gate in ID.
The ID cubic model already predicts physical increments to approximately 1e-7;
test the reference mapping, not another fit, weight, seed or environment sweep.

Use existing seed21/22 on evaluation JSON. Project its fitted linear/cubic action
matrices into its fitted load matrix to obtain positive componentwise actuator
coefficients a/c. Reuse its local gain K. Candidate policy is the unique monotone
inverse of a*u+c*u^3 applied to `(a+c)*tanh(-diag(a)*K*z/(a+c))`. Its action
Jacobian at z=0 remains -K, and the force cap comes from a+c. No oracle plant
matrices or actuator coefficients enter this candidate; no tunable scale is added.

Compare fitted-local, fitted-inverse and the existing known-inverse diagnostic
control on fresh paths 63000--63099, radii 0.2/0.5/1.0, severity 1/50, horizon 200.
Keep plant, load process, observation and physical cost unchanged. First reproduce
fitted-local and known-inverse on the first two v31 paths in every block.
Report four-window state/command costs, saturation, state norm, paired path cost
changes and initial-state action/physical-force Jacobians.

Formal: two CPU tasks, one per existing training seed; 3,600 fresh controller
episodes plus 48 reproduction episodes = 729,600 transitions. Numerical preflight
uses seed21 and two fresh paths per block: 60 episodes / 12,000 transitions.
All tasks have zero fitting, RL training, optimizer updates and checkpoint access.

Admit one subsequent reference-only learner intervention only if fitted-inverse
strictly reduces mean total cost in every seed/radius/severity block. No seed
pooling or rule changes after outcomes. This gate tests the reference only;
v31's learner gates and all historical HOLD decisions remain unchanged.

Use any free compatible Linux/WSL CPU node, no preference/pin/allowlist or Windows.
Inputs and outputs are compact JSON; no actor, replay, checkpoint or raw trajectory
transfer. This does not establish an actor-training benefit or general RL superiority.
