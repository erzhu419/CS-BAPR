# v64 conditional signed-vector screen

Keep the frozen v60 dynamics, normal residual normalization, all11 coordinates,
16-transition residual/risk histories, mild profiles and beta=-2. Change the
normal reference geometry using observed initial/after-reset episode phase and
episode age. Use existing16-step age knots, radius8 and relative covariance
ridge1e-3. Interpolate centers and positive-definite precision matrices.

The two saved v62 normal-fit trajectories (seed134000, frozen/updating6000
steps each) fit age/phase centers and covariances. The two separate normal
calibration trajectories (seed134010) fit global95th-percentile score scales
and the existing phase/age95th-percentile departure thresholds. Compare center
only with unchanged global precision against center plus local precision.
No OOD trajectory fits parameters or chooses a threshold/window/ridge.

Replay the12 v62 tests and five v61 development trajectories. Inherit the three
unchanged control readouts from exact v63 results. Keep groups separate and
normal first800/full/last1000/last64 means <=0.10; every changed first50 and
tail64 contrast must remain >0. Full center/covariance is the predefined main
candidate; center-only is an ablation, not automatic fallback selection.

This is retrospective screening with0 new interactions/updates. Persist the
new reference and matrices only on node007; return compact scalar JSON/logs.
Observed episode boundaries choose phase; true mode and switch tick are only
offline labels. Independent new continuations follow a successful screen.
