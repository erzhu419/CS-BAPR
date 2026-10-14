# v102: local mean/covariance/history attribution

Use the three new v101 nonpositive posterior-margin windows: seed135191 high_grav early, seed135192 heavy early and high_grav tail. Retain each matched frozen path and both normal controls: 12 windows, 656 transitions, 10 distinct paths. Keep v98 as incumbent.

Cross the incumbent and age-balanced conditional means/covariances (four combinations), then initialize each from either recipe's exact pre-window posterior (eight combinations). Reuse native full-path likelihood caches to recover the prefix; posterior state persists across episode resets exactly as before. Diagonal density, cached likelihood slices, posterior statistics and prediction RMS must reproduce the complete replays: 72 comparisons of each type. A two-window smoke checks these before the full diagnostic.

Reuse all fitted errors and native reference payloads. No new simulation, fitting, calibration, tuning or actor feedback. Only scalar JSON is synchronized. Report instantaneous energy/volume contributions separately from prefix-history sensitivity; do not use the crossed branches as calibrated risk or hybrid-policy evidence.

## Limitations

Failure-selected development windows, one source policy and known modes. Crossed histories change only the incoming posterior; they do not reconstruct an entire counterfactual history under mixed models. Attribution is local to the specified windows, not RL-return or independent confirmation evidence.
