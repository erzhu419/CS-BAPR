# v62 causal residual innovation

Freeze the v60 long-normal dynamics, residual normalization, level covariance
and level scale. Add the difference between adjacent16-transition mean residual
vectors. Its zero-centered normal second moment uses all11 coordinates and the
existing1e-3 relative covariance ridge. Before32 observations its score is0;
both window histories persist across actual episode boundaries.

New normal fit seed134000 estimates innovation covariance; new calibration
seed134010 estimates its95th-percentile scale and the initial/after-reset age
curves. Both include frozen and updating6000-step branches. Joint signal is
log1p(max(level/level95, innovation/innovation95)); its threshold is calibrated
jointly, not the maximum of independently thresholded risks. Compare frozen
v60, level-only with identical new calibration, and level-plus-innovation.
No switch tick, mode label, future observation or v61 diagnostic anchor enters
the readout. Keep the existing16-step risk smoothing.

Test seeds134020/134021 each have a frozen normal branch and five paired
updating normal/heavy/slippery/high_grav/compound branches. All6000 steps,
switch3800, unchanged mild profiles, stochastic actions, fixed beta=-2. Total
16cells/96000 interactions/72000 updates. Source seed1, updates99745,
shared replay6144 and context100000 remain matched.

Keep normal first800/full/last1000/last64 means <=0.10 and every changed
first50/tail64 contrast >0. No independent-source or adaptive-return claim.
Retain raw traces and resumable native/replay/RNG state server-side only;
return scalar JSON/logs. If this preflight passes, test the frozen causal
readout before reopening closed-loop RL comparisons.
