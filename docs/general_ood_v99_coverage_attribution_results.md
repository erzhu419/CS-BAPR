# v99: coverage remains incomplete; HOLD

Corrected native task t140751 completed successfully in 52s: 20 windows, 1084 transitions, 120 diagonal density comparisons and 120 prediction-RMS comparisons. Six local tests pass. Initial task t140735 failed after 18 windows because the two historical v62 traces lack oracle observations; the repaired diagnostic preserves support/density results and leaves their direction metrics unavailable.

Six remaining v95 failure windows (seeds135191/135192) have 0--0.0244% v97-extension neighbors, with over 99.48% from seed135050. Their means barely change between recipes and covariance substitution slightly worsens instantaneous best-other margins. The other remaining failure, seed135190 compound early, has 81.87% new neighbors but still fails: coverage is not the only limitation.

For repaired seed135190 slippery/high_grav early windows, symmetric instantaneous-margin contributions are mean +0.073/+0.253 and covariance +0.356/+0.914. These are fixed-window density effects, not a causal decomposition of complete posterior histories.

Next: equal-budget mean/support pilot on unused normal-prefix references versus incumbent OOD suffix and an age-balanced mixture. Reuse native v97 trajectories; no further collection, covariance tuning or actor integration.

## Limitations

Failure-selected development evidence from one source policy and known modes. Crossed scores are not recalibrated belief or RL-return evidence. The two v62 direction metrics are unavailable. Preserve v98's conditional coverage gains without claiming an added-disagreement benefit.
