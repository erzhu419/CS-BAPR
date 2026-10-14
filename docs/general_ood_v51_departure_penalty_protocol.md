# v51: normal-age departure penalty

Keep v50 frozen policies, dynamics model, references, original log signal,
change-point filters, physical profiles and paired trajectories unchanged.
Fit normal-only q95 log-signal thresholds at episode-age knots every16 steps,
using neighborhoods +/-8 steps; interpolate linearly between knots. No OOD data
fit or tuning. Beyond the observed maximum age, hold the endpoint threshold.

Instantaneous departure is 1-exp(-max(signal-threshold(age),0)); its causal
last16 physical-step mean is the candidate coefficient. Hypothetical beta=-2
-coefficient. Never clear its history at reset or mask observations. Compare
age-departure with global-departure (threshold=log2) and the unchanged v50
noise-likelihood age readout. Quantile values are not probability certificates.

Retain five v50 branches per development seed and first16/50/256/last64 readouts.
Add normal-only seeds99300/1 in preflight and90100--90119 in formal, disjoint from
fit/calibration/development probes. Record whole-run, initial and every complete
first50 restart window for reused/new normal controls. Old v50 records must match
exactly. All arms share each physical trajectory; no action or optimizer feedback.

Preflight 67584 interactions; formal 503808 (24 references, 240 tests). Checkpoints
stay on node007, scheduler selects idle GPU without preference; no Windows nodes.
Return scalar JSON/logs only. Keep failures, no parameter changes after preflight.

After null behavior and changed-pair response are inspected, decide whether a
matched-start RL pilot is justified. V40 snapshots have optimizer state but no
replay: such a pilot requires a shared replay refill, not a claimed exact resume.
No training seed expansion or manuscript changes in this round.
