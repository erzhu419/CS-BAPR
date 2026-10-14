# v106 expanded standard mean with frozen incumbent covariance

v105 admits only the standard reference expansion: all six coverage failures
improve, while action perturbation loses the matched-budget comparison.

Use 34000 mean rows from ten collection sources, with the exact complete v98
covariance built from its original eight sources: 27400 raw rows, 29600 error/
covariance-mean rows, old member predictions and old mean-centered displacement
moment. Keep forward models, scales, mixture weight, hazard and coefficient fixed.
The new mean has separate group membership; it does not create covariance members.

Native smoke checks all three covariance conditions against the incumbent,
physical-unit Gaussian scores and v105 prediction RMS. Then recalibrate on the
same 12000 normal-null transitions with the same phase selection/populations.
Only the mean-row count is allowed to change in the calibration protocol.

Replay all 95 existing paths from the beginning (570000 transitions), retaining
recursive history and episode-reset behavior. Reuse the unchanged v94 controls,
verify the 186 v105 prediction RMS comparisons, and examine six hard windows,
all updating/frozen modes, normal windows and historical slippery behavior.
Existing per-path caches and progress support continuation. No new interactions.

## Limitations

This is a development intervention with larger reference budget than v98 and a
frozen, rather than newly fitted, covariance. Gains in prediction do not ensure
mode recognition or return gains. Normal local regressions require explicit
normal-risk assessment. No actor integration or manuscript-table change.
