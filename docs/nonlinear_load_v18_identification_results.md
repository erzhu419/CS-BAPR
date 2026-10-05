# Nonlinear load v18: actuator-aware identification passes

Tasks t126814/t126815 exited0 on automatically selected jtl311linux GPU0/GPU1, in about80 seconds each including runtime setup. Original fits and existing controllers reproduced. All three frozen gates passed; 4,800 controller episodes and zero learner updates. Only1,882,696 bytes of result JSON were pulled, without trajectories or checkpoints.

## Main Result

- Independent ID physical RMSE: seed11 0.0133025 -> 0.000002729; seed12 0.0132281 -> 0.000001876.
- Independent OOD physical RMSE: seed11 0.0169205 -> 0.000295355; seed12 0.0190987 -> 0.000333435 (about98.25% lower).
- Load-gain maximum absolute error: seed11 0.165098 -> 0.000249655; seed12 0.319133 -> 0.000250043. Cross-axis gains disappear or fall to numerical scale.
- Direct-feedback costs decreased in every block: ID about1.82--2.18%; OOD seed11 about5.26--11.19%, seed12 about13.05--27.84%. This supports omitted cubic action terms as a material source of the previous reference errors.

## Next Experiment

Freeze a fresh matched actor cohort on seeds13/14: original linear reference versus cubic-fit local reference, two arms per seed. Keep v17 reset-balanced Jacobian sampling, weight0.01, architecture, severity1 training,500 episodes and last-checkpoint evaluation unchanged. Both references use the same per-seed identification transitions; the only intervention is the action library and its fitted local derivative. Evaluate both loads and all radii on the existing paired paths, including each actor's own fitted reference. Require radius1 total/early-state improvement, radius0.2 retention and all-radius own-reference added value per seed/load before learner-comparator replication. No new training was submitted in v18.

## Limitations

One true positional load coefficient0.00125 remains zero in each cubic fit, leaving nonzero OOD model error; do not retune the frozen threshold. Radius1 ID direct-feedback early state cost increased about0.0226--0.0227 despite lower total cost. Direct-reference improvement is not actor improvement or an optimality certificate. Historical v13/v15/v17 HOLD and paper tables remain unchanged.

Evidence: `results/nonlinear_load_v18_identification/{task_observations,summary}.json` and both per-seed diagnostic JSONs. Reproduce with `python3 scripts/summarize_nonlinear_load_v18.py`.
