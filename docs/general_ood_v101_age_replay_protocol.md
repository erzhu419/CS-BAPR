# v101: complete age-balanced density and belief replay

Promote only v100's fixed age-balanced recipe: normal steps1600:3800 at stride2 and high_grav/compound steps3800:6000 at stride4, for each native v97 seed135200/135201. Each seed supplies 2200 inputs with five stored counterfactual targets. The incumbent v98 diverse_recipe suffix replay is reused as the equal-budget comparator; no duplicate reference fitting or replay.

Keep 27400 raw rows, 29600 mean/error rows, eight source groups, original forward models/scales, 64-neighbor ridge alpha0.064, raw weight and disagreement coefficient1 fixed. Recompute every error row after excluding its complete source seed from all mean rows. Calibrate all three covariance controls on the same 12000 normal-null transitions with unchanged phase populations, posterior hazard and temporal rules. Reuse the tested v98 error/phase/term/progress cache machinery; native caches retain its implementation revision and carry the distinct age_balanced recipe in a new directory.

First run a native smoke (16 partial held-out error rows, three unchanged-v94 density comparisons and four v100 mean matches). Admit fitting only after smoke success; admit complete replay only after all error folds, physical-density checks and matched calibration pass. Replay all 95 development paths/570000 transitions, retaining normal, frozen/updating and historical controls, complete posterior history and reset semantics. Compare all three covariance conditions, per-seed gains, failure windows and normal mean risk with the incumbent; do not equate mean risk with false-positive rate.

Only the candidate needs new tasks. Fits and replay share one native checkpoint directory; fits are resumable through completed folds/terms and replay through progress.json. Code is frozen before first staging. Only scalar JSON is synchronized; all trajectories, arrays and caches stay on their existing server.

## Limitations

One source policy and known-mode simulator references. These already inspected trajectories provide development evidence, not independent confirmation or RL-return gains. No new environment steps, actor feedback, coefficient tuning or seed expansion.
