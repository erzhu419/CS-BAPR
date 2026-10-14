# v56: localize v55 gravity-tail suppression

v55 formal remains HOLD: 16/16 positive early contrasts, 15/16 positive tails.
Replay its two normal learning calibration cells (98700/1) and the failed
gravity98723 cell, each1200 steps; reuse the saved normal98723 control. Added
cost3600 interactions and3600 updates, no new confirmation seeds.

Use the frozen v55 runtime/source/replay, v53 dynamics, v55 formal reference,
beta=-2, stochastic actions and step800 switch. Require exact recorded scalar
cell metrics, paired prefixes and both saved candidate control/change readouts;
reproduce age calibration within1e-12. The recomputed reference only verifies
the old calibration and never replaces the readout reference.

Report normal calibration q95/max separately for initial and later episodes at
fixed age knots, plus paired signal/threshold/exceedance/coefficient statistics
before/after the actual reset and in the original windows. Do not select a new
threshold, tune gates, change the policy or infer RL gains from this replay.
Save full progress/RNG/replay/native state every500 steps and between cells.
Only scalar JSON/logs return; raw traces/checkpoints remain on node007.
