# v107 cached mean/calibration attribution

v106 improves recognition but raises the worst updating normal-window mean risk
from 0.00438 to 0.03205. The existing 12000-transition normal calibration already
contains frozen and updating trajectories, both from seed135060.

Cross the cached v98/v106 likelihoods with the saved v98/v106 calibrators (four
arms, all three covariance conditions). Keep complete trajectory posterior and
departure history, including episode boundaries. Calibration must not change
posterior scores or mode accuracy. Reproduce both original diagonal results.

Analyze 18 paths: all six v95 updating/frozen normal paths, five original
negative-window mode paths and their frozen counterparts, and the matched
v62 seed134020 normal/slippery updating pair. Report normal risks, recognition,
changed-path risks and normal episode-level signal/threshold diagnostics.
Smoke uses all four seed135192 normal/compound updating/frozen paths, so both
groups have normal and changed populations for the summary.

Reuse native likelihood and phase caches; no density geometry, model fitting,
calibration fitting, new environment interaction or actor integration. Sync JSON only.

## Limitations

These are selected development paths. Old-calibration crossovers are diagnostic,
not a validated fix. Calibration cannot repair posterior mode errors, and window
mean risk is not a false-positive rate. No candidate adoption in this round.
