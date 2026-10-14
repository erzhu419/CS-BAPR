# v102: covariance dominates the three new failures

Task t140940 completed in 109 seconds. All 12 windows (656 transitions), with matched normal/frozen controls, passed 72 comparisons each for density, cached likelihood slices, posterior statistics and prediction RMS. Exact incoming histories were recovered from native full-path caches; only scalar JSON was synchronized.

For suffix_disagreement_mix, six-order contributions to the change in mean posterior margin are:

| Updating window | History | Mean | Covariance | Total |
| --- | ---: | ---: | ---: | ---: |
| seed135191 high_grav early | -0.021 | -0.021 | -5.915 | -5.958 |
| seed135192 heavy early | -0.009 | +0.066 | -10.614 | -10.557 |
| seed135192 high_grav tail | -0.488 | +0.610 | -7.022 | -6.899 |

Using age-balanced means with incumbent covariance and incoming history gives accuracies of 96%, 92% and 67.19%, versus incumbent 96%, 92% and 53.13%, and age-balanced 26%, 32% and 12.5%. New covariance also strengthens the matched updating normal margins by about 7.2-8.6; frozen mode/normal accuracies remain 100%.

Next: decouple mean support from covariance construction, freezing the complete v98 covariance geometry, including its disagreement moment. Recalibrate normal risk and replay all 95 complete development paths. This follow-up is not dispatched; retain v98 and HOLD actor integration. Paper tables are unchanged.

## Limitations

Failure-selected development windows, one source policy and known modes. Local substitutions are not a complete hybrid-history replay, calibrated-risk result or RL-return improvement.
