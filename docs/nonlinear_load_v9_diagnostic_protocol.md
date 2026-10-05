# Nonlinear load v9: read-only diagnostic

Use the existing final candidate checkpoints from Linux tasks t120342 (seed5) and t120346 (seed6), with the frozen training runtime. No optimizer updates, refits, model writes, new training seeds, environment changes or gate changes. Run CPU inference at checkpoint locations; this is data-local diagnosis, not a training placement preference. Transfer only the diagnostic script and small result JSONs, never checkpoints.

Compare actor, saved fitted feedback, frozen known local-linear and frozen inverse-actuator controls at severities 1/20/50, on all 100 paired paths 50000--50099. Decompose each path's total cost into physical-state cost and command-effort cost using the environment's pre-step float64 state. Check their sum against rewards and reproduce all saved returns with rtol=1e-4, atol=1e-3 before accepting results.

Sample actor trajectories every ten steps (2,000 states per severity); compare all four controls on these same states. Report normalized-action and physical-output-force Jacobians separately, including physical-state and current-load input blocks. Use the actuator chain rule, with the monotone inverse solved by the existing SciPy root finder. Include origin feedback matrices, all-state metrics and the largest-load quartile. The identity-plus-output-force/load-Jacobian residual measures instantaneous cancellation, not optimality under command costs.

Retain full 100-path cost arrays in small JSONs, mean paired cost decompositions and the five largest actor excesses. Do not select successful paths or replace the failed v9 added-RL-value gate. Interpret these diagnostics before choosing a training ablation; manuscript tables stay unchanged.
