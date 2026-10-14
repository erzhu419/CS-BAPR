# v71: equal-label state-coverage experiment

Keep the v69 five-mode quadratic likelihood bank, Gaussian/OAS calibration,
16-step readout, normal-null threshold rules and frozen source actor unchanged.
Both conditions retain all v69 fitting/density-calibration data (12,000 labels
per physical mode and role). Each adds 6,600 labels per mode and role:

- normal-only: post-3,800 states from three new normal paths;
- shifted-state: the paired normal, high-gravity and compound paths.

Fitting seeds 135080/81/82; independent density-calibration seeds 135090/91/92.
Collect the normal partner of each changed path, verify identical prefixes,
and generate all five physical labels at every full simulator state. Only the
2,200-step suffix enters the extension. Same label count, model capacity,
scaling procedure, ridge and readout rules. Each bank recalibrates its null on
the same untouched v69 normal-null streams, not on test data.

Freeze both references BEFORE collecting test seeds 135120/121, each with
normal/gravity/compound paths. Separately replay v70 and old v61/v62 traces;
none enter fitting/calibration. Preserve the existing normal mean-risk <=0.1,
positive first-post50/last64 paired contrast and positive true-mode-margin
requirements. Do not choose parameters or connect belief from this screen.

16 paths x 6,000 actual x 6 simulator calls = 576,000 total steps; zero new
optimizer updates. Raw arrays and checkpoints stay remote; scalar JSON/logs
only. Linux eligible inputs currently live on node007; no node/GPU preference.

## Limitations

Known mild profiles, privileged simulator labels, one source training seed.
Coverage in fitting and density calibration changes together. New frozen-path
diagnostics are not old updating-path reproduction or closed-loop confirmation.
