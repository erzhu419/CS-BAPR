# v65 state/action conditional reference

Keep the frozen v60 dynamics, all 11 residual coordinates, causal 16-transition
residual/risk histories, beta=-2 and the existing physical profiles. Condition
normal residual moments on the causal 16-transition mean of 11 observations
and three executed actions. The next state enters the residual, not the
conditioning inputs. Episode age/phase only selects the existing scalar
departure threshold; it does not select the vector center/covariance.

Use the two saved v62 normal-fit trajectories (seed134000, frozen/updating,
6000 transitions each). Retain endpoints of disjoint full 16-step windows:
750 normal reference windows. Standardize conditioning inputs using these
fit windows and use 64 nearest windows for a local residual mean/covariance.
The fixed neighborhood provides about six windows per residual coordinate;
OAS covariance shrinkage estimates its weight from these normal neighbors.
Keep the existing relative ridge1e-3. No neighborhood/threshold/window search.
Use SciPy cKDTree and sklearn.covariance.oas ([OAS documentation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html)); neither estimates geometry from
the query residual. Compare conditional center with unchanged global precision
against conditional center plus OAS local precision (predeclared main).

The separate saved normal calibration trajectories (seed134010, frozen/updating,
12000 transitions total) fit global95 score scales and phase/age95 departure
thresholds. Neighbor radius95 is diagnostic only and does not control risk.
Residual and input histories persist across observed episode boundaries.

Replay all 12 v62 tests and five v61 development traces, with no new RL
interactions/updates. Inherit five v64 control readouts unchanged and keep
groups separate. All frozen/updating normal first800/full/last1000/last64
mean risks must be <=0.10; every changed first50/tail64 contrast must be >0.
Advance to independent continuations only if the main candidate passes both
groups. Keep full references/arrays on node007 and return scalar JSON/logs.

## Limitations

These are reused development trajectories from source training seed1, not
independent confirmation. Disjoint residual windows can still be correlated;
OAS regularizes covariance but does not guarantee conditional coverage. The
readout does not control actions in this screen, so it cannot establish RL gains.
