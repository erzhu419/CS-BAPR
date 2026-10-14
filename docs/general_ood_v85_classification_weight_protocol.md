# v85 Classification-loss weight pilot

Reuse the six v83 cached whole-seed geometries and the v84-verified 25 true/candidate offsets per input. Preserve local-linear means, raw/cross-seed covariance components, scaling and all temporal rules. Change only the shared raw-covariance weight's selection objective to uniform-prior five-mode classification NLL, averaged equally over seeds, true modes and rows.

Use the existing bounded scalar optimizer plus both endpoints and the frozen v83 density-selected weight. This is a numerical candidate comparison, not a global-optimum certificate. Reproduce v84 classification losses at the three old weights before selection; a mismatch requires correcting the cache/scoring correspondence.

Report aggregate, per-fold and per-mode probability loss, accuracy, signed margins and confusion fractions; retain all three controls. Inspect the localized v84 regressions (135171 normal/slippery, 135050 slippery, 135092 heavy) before deciding whether another query replay is informative. No evaluation paths, normal-null calibration, fresh seeds, actor updates or paper tables are included. Only scalar JSON/logs return locally; banks/caches remain on node007 with the native runtime.

## Limitations
The same reference folds select and diagnose the weight; they are not independent confirmation. Outer labels leave mean/covariance fits but enter weight selection. Instantaneous classification does not establish recursive query accuracy or RL return improvements.
