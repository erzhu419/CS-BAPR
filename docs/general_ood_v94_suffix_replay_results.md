# v94 Paired suffix uncertainty replay

Task `t139195` completed on node007 CPU, exit 0, in 4,942.99s (1h 22m 23s); results synced successfully. All 65 paths (48 changed, 17 normal) and 96 changed windows are present. All five reused controls match v90 at the record level. Each new readout used the same 12,000 normal-null transitions. Only the 3,787,592-byte scalar JSON returned; traces and caches remain server-side.

| Readout | Non-positive recursive margins / 96 | Equal-window accuracy | Maximum normal-window mean risk coefficient |
| --- | ---: | ---: | ---: |
| v78 raw covariance | 14 | 84.54% | 0.007962 |
| v83 selected mixture | 12 | 87.06% | 0.007605 |
| v90 prefix mean, frozen errors | 9 | 89.50% | 0.005815 |
| Suffix matched error support | 12 | 86.35% | 0.006649 |
| Suffix directional disagreement | 7 | 89.41% | 0.005452 |
| Suffix trace-matched scaling | 11 | 86.59% | 0.004961 |

Directional disagreement resolves five negative windows against the no-disagreement control and four against trace-matched scaling, introducing none against either. Accuracy improves/regresses/is unchanged in 34/8/54 windows against the former and 29/5/62 against the latter. Early/tail mean accuracy is 89.04%/89.78%, versus 86.79%/85.90% without disagreement. All 96 paired departure-risk contrasts remain positive; five of eight groups pass the existing screen, with v76/v62/v61 still failing.

The new conditions have identical true-mode prediction RMS. Their directional and scaled covariance traces match to a maximum absolute difference of 1.3643e-12 over all evaluation paths. The paired advantage therefore supports a useful directional covariance contribution beyond trace inflation, without changing the conditional mean. It is not a mean-prediction improvement.

- v62 seed134021 heavy tail: accuracy 26.56% -> 90.63% (scaling: 25.00%), recursive margin -17.19 -> 12.32. Against fixed competitor high_grav, energy advantage changes -17.10 -> 5.92 while volume advantage remains about 2.56. This repair mainly changes precision-weighted error ranking.
- The other resolved windows are v62 seed134020 heavy/high_grav tails, v62 seed134021 slippery early and v76 seed135181 heavy tail. Positive average margins are not perfect recognition: v62 seed134020 heavy/high_grav tail accuracies remain 45.31%/43.75%.
- The seven remaining negative windows are v76 seed135180 slippery/high_grav/compound early, v62 seed134020 slippery tail, and v61 seed132020 slippery tail plus high_grav early/tail. v76 slippery/compound early accuracy is 8%/12%; v61 slippery and high_grav tails are both 0%. In the v76 slippery early fixed-normal comparison, the volume disadvantage remains about -4.18 while energy advantage falls 3.42 -> 2.11, worsening the margin.
- Against v78, nine negative windows resolve but two appear: v61 slippery tail and v76 seed135180 high_grav early. Against v90 prefix, four resolve and two appear (v61 high_grav early/tail); overall accuracy is 0.0892 percentage points lower. The candidate is not uniformly better than these older alternatives.

**Decision:** HOLD; retain v78 as the working baseline and freeze `suffix_disagreement_mix` as the candidate for new-environment-seed confirmation. Keep the prefix result as complementary development evidence. No coefficient tuning, actor integration or paper-table replacement.

**Next:** compare the frozen v78, suffix support, directional and trace-matched readouts on paired trajectories from at least three previously unused environment seeds, with the same normal/heavy/slippery/high_grav/compound modes and frozen/updating policy settings. Freeze the reference bank, full-bank errors, member fits, weight, coefficient and phase calibration before generating those paths. Report the known old counterexamples separately rather than replacing them. Assess negative margins, equal-window accuracy, normal-window risk and mode-specific regressions before any closed-loop RL experiment. Dispatched as v95 tasks `t139551`/`t139552`/`t139553` for seeds135190/135191/135192 after native preflight `t139549` passed; references and phases are loaded without refitting.

## Limitations
These are inspected development paths from one source training seed and known mild modes, not independent algorithm confirmation or RL return evidence. Accuracy equally averages 48 changed paths and two windows; the normal statistic is a risk coefficient, not a false-positive rate. Source-deletion sensitivity is not calibrated variance; equal trace does not equalize determinants or precision. New environment seeds would test trajectory generalization conditional on this source policy, not replication over independent training seeds.

Result: `results/general_ood_v94_suffix_replay/replay/suffix_replay.json`.
