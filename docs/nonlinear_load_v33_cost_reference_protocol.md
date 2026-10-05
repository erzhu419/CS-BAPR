# v33: cost-aware nonlinear reference

Frozen 2026-10-04 (Asia/Shanghai), before v33 rollout outcomes. v32's actuator
inverse reduced OOD command cost but increased state cost and worsened every ID
block. Preserve that failure; test the actual control objective, without tuning
weights, modifying the environment or training actors.

Reuse seed21/22 ID cubic fits. Physical next state is the fitted A*x+D*f+B1*u+B3*u^3.
The conditional mean of next observed load is (1-refresh_probability)*f; the
refresh probability is the documented environment value, not an inferred future
draw. Form this six-state local-linear model and solve its undiscounted discrete
Riccati equation with the unchanged state cost and command cost scale^2*R.
Its value P is a terminal approximation. Choose u in [-1,1]^2 minimizing
u^T R_command u + next(z,u)^T P next(z,u), with the ID cubic actuator retained.
The fitted axes are separable: enumerate all real roots of the quintic
stationarity equation and both bounds using NumPy's polynomial solver.

Controls: old fitted-local, v32 fitted-inverse, known-inverse, and cost-linear
(the same augmented Riccati gain, clipped to the same bounds). Cost-linear and
cost-nonlinear match at their origin Jacobian; the load columns may differ from
the old full-cancellation gain. This separates a cost-model correction from
nonlinear added value. No oracle plant matrices enter either cost candidate.

Fresh paths 64000--64099, radii 0.2/0.5/1.0, severity 1/50, horizon 200. First
reproduce all three existing references on the first two v32 paths per block.
Report per-path/four-window state and command costs, saturation/state norms,
and solver objective comparisons with cost-linear under the nonlinear model.
Formal: 6,000 fresh plus 72 reproduction episodes = 1,214,400 transitions,
two CPU tasks. Seed21 numerical preflight: 96 episodes / 19,200 transitions.
Zero new identification, training, optimizer updates or checkpoint access.

Admit one reference-only learner intervention only if cost-nonlinear strictly
reduces mean total cost versus both fitted-local and cost-linear in every
individual seed/radius/load block. No pooling, post-outcome tuning or automatic
learner replication. Runtime NumPy 2.2.6/SciPy 1.15.3, Python 3.10/3.11; any free
Linux/WSL CPU, no placement preference or Windows. Compact JSON transfers only.

## Limitations

The one-step Riccati surrogate is not the exact finite-horizon nonlinear value;
its global action minimizer is not an optimal-policy certificate. Derivatives
apply on a unique interior minimum or a stable saturated branch, not at branch
ties. The existing two ID fits do not constitute independent learner replication.
Historical HOLD, v31/v32 failures and the manuscript remain unchanged.
