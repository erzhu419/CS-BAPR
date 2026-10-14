# v106: expanded standard mean, frozen covariance

Task `t141081` completed successfully on node007 in 2h30m03s. The matched
normal calibration and 95-path replay covered 570,000 cached transitions.
Unchanged controls, complete incumbent covariance and 186 v105 prediction-RMS
comparisons passed. No environment interactions or actor updates were added.

Primary condition: `suffix_disagreement_mix`, equal early/tail window weights.

| Population / metric | v98 | v106 |
| --- | ---: | ---: |
| v95 updating recognition accuracy | 68.12% | 74.27% |
| All updating recognition accuracy | 81.39% | 84.69% |
| Six old failure windows: accuracy | 14.44% | 30.55% |
| v95 updating nonpositive-margin windows | 7 | 6 |
| Historical slippery tail: accuracy | 32.81% | 65.63% |
| v95 frozen recognition accuracy | 98.75% | 98.75% |
| Maximum updating normal-window mean risk | 0.00438 | 0.03205 |

The normal-risk regression is concentrated on updating seed135192: full-history
normal-mode accuracy falls from 98.58% to 95.22%. Calibration alone cannot repair
this posterior classification loss. The original normal calibration already
includes both frozen and updating trajectories (seed135060, 12,000 transitions).

**Decision:** retain v98; do not connect this readout to the actor. v107 crosses
cached old/new likelihoods with old/new calibrators to attribute the regression.

**Limitations:** development-only paths and one source policy; the expanded mean
uses 8,800 rather than 4,400 extension source-index union rows. These risk means
are not false-positive rates, and recognition gains are not return gains.

Artifacts: `results/general_ood_v106_standard_mean_replay/replay/summary.json`
and `regime_replay.json` in the same directory.
