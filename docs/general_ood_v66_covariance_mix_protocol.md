# v66 normal-block covariance mixing

Retain v65's normal state/action window index and conditional centers, frozen
v60 dynamics, 64 neighbors, local OAS covariance with ridge1e-3, 16-step histories,
beta=-2 and all environment/gate settings. Change covariance to
`weight*global_covariance+(1-weight)*local_covariance`, where the global endpoint
is exactly inverse(v65 global precision) and the local endpoint is v65's
regularized OAS covariance. Select one global weight, not mode-specific weights.

Select the weight from v62 normal-fit traces only: seed134000 frozen/updating,
6000 transitions each. Six coupled 1000-transition holdouts exclude the same
time block in both branches. A training window is excluded if any of its16
transitions touch the heldout block. Validation uses full disjoint16-step
windows entirely inside the block. Refit input scaling/neighbor centers and
local covariances on each fold's training windows. The inherited global endpoint
comes from earlier normal data, not these validation blocks.
Each fold retains624 training windows and124 validation windows, and excludes
two boundary-crossing windows. The six folds provide744 validation windows.

Minimize heldout normal Gaussian negative log likelihood with
[SciPy bounded scalar optimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html),
also compare its solution with both endpoints. This is a normal-data fit, not
an OOD threshold/parameter sweep. Freeze the selected weight before evaluating the
separate seed134010 calibration traces. These12000 calibration transitions
fit score95 and the same phase/age95 departure thresholds.

Replay all17 existing v62/v61 cells, inherit all seven v65 controls, and keep
groups separate. All frozen/updating normal first800/full/last1000/last64 means
must remain <=0.10, and every changed first50/tail64 contrast >0. No fallback
ablation selection. Added RL interactions/updates0. Node007 CPU reads data
in place; only scalar JSON/logs return, full references stay server-side.

## Limitations

The bounded solver is a local optimizer. Normal predictive likelihood need not
optimize OOD detection. Blocked validation is not independent environment-seed
confirmation; reused trajectories and fixed beta establish no adaptive RL gains.
