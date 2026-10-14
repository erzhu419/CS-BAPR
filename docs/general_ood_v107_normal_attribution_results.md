# v107: normal regression comes from changed likelihoods

Corrected native smoke `t141228` passed; formal task `t141229` completed in
102 seconds. All 18 paths / 108,000 cached transitions and 108 diagonal
path-condition comparisons passed, preserving complete posterior histories.

Updating normal seed135192, `suffix_disagreement_mix`, full-history mean risk:

| Mean | Calibration | Mean risk |
| --- | --- | ---: |
| Old | Old | 0.004383 |
| Old | New | 0.004438 |
| New | Old | 0.032003 |
| New | New | 0.032052 |

The same pattern holds for support and trace-match covariance. Calibration
does not change posterior classification. Normal-mode accuracy falls from
98.58% to 95.22%; compound predictions rise from 0.033% to 2.85%.
The largest episode regression is steps [2000, 3000): accuracy 98.8% to 81.4%,
mean risk 0.00183 to 0.14387. Other updating normal seeds do not show this loss.

**Decision:** retain v98; changing only calibration is not a repair. Next,
attribute normal-versus-compound likelihood changes with mode-wise mean
ablations, including old normal mean plus expanded OOD means and its reciprocal.
Use cached likelihoods and matching cached normal-null calibration inputs;
no new trajectories or covariance computation are needed for this diagnostic.
Follow-up: v108 mode-wise mean ablation; see
`docs/general_ood_v108_modewise_mean_protocol.md` and its dispatch manifest.

**Limitations:** selected development paths, one source policy, and known modes.
The old-calibration crossover is diagnostic, not validated calibration for the
new density. Risk means are not false-positive rates; no actor or return claim.

Artifact: `results/general_ood_v107_normal_attribution/formal/normal_attribution.json`.
