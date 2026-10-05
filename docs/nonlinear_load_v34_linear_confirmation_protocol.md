# v34: independent cost-linear confirmation

Frozen 2026-10-04 (Asia/Shanghai), before v34 outcomes. The cost-linear secondary
control improved all 12 blocks in v33; its nonlinear primary candidate failed.
Confirm this new hypothesis on unused paths without changing any controller.

Reuse seed21/22 ID cubic fit JSON and the exact v33 CostReference.linear_actions:
u=clip(-K_cost*z,-1,1). K_cost uses the existing physical fit, unchanged physical
state and command costs, and documented load persistence 1-refresh_probability.
Reproduce the saved v33 gain and all four controllers on its first two paths in
every block. Controls: old fitted-local, cost-linear, fitted-inverse, known-inverse.
No nonlinear action optimization, new fit, actor training or checkpoint access.

Formal paths 65000--65099; numerical preflight uses separate paths 65500--65501.
Same radii 0.2/0.5/1.0, severity 1/50, horizon 200 and four 50-step cost windows.
Two formal CPU tasks: 4,800 new plus 96 reproduction episodes, 979,200 transitions.
Seed21 preflight: 96 episodes / 19,200 transitions. Transfer compact JSON only;
reuse immutable dependencies. Any free compatible Linux/WSL CPU, no preference,
pin, allowlist or Windows.

Primary gate: cost-linear must strictly reduce mean total cost versus the old
fitted-local reference in every individual seed/radius/load block. Comparisons
with fitted-inverse and known-inverse are descriptive and remain reported.
Passing admits only a single reference-only learner intervention, not seed or
baseline expansion. Failing stops this candidate without tuning or gate changes.
Preflight is numerical admission; it does not enter the performance decision.

## Limitations

This confirms reference performance, not actor learning or an optimal-controller
claim. The two existing ID fits are nearly identical, not independent learner
replications. v31/v32/v33 failed gates, historical HOLD and the manuscript remain
unchanged regardless of the v34 reference decision.
