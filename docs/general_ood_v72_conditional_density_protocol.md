# v72: frozen-forward conditional residual density

Freeze the v71 shifted-state forward coefficients, scaling, ridge and physical
profiles. Compare its global residual Gaussian to two 64-neighbor variants:
conditional residual mean with global covariance, and conditional residual
mean with local OAS covariance. Reuse v65/v66 neighbor count, OAS and relative
ridge. Condition only on the current state/executed action; retain the Gaussian
constant, covariance log determinant and physical target-scale Jacobian.

Geometry uses the same 18,600 v71 density-calibration labels per physical mode;
no evaluation labels or new forward fitting. Recalibrate candidate departure
curves on the same two untouched v69 null streams, keeping the 16-step readout
and threshold rules. Freeze all references before collecting seeds 135130/131.

Fresh normal/gravity/compound branches: 6,000 steps, switch 3,800, TimeLimit
1,000, stochastic source policy, causal context, zero optimizer updates. Exact
true-mode shadows and paired prefixes must pass. Replay v71/v70/v61/v62 as
development groups. Keep every normal risk <=0.1 and positive first50/last64
paired risk contrasts AND true-mode margins as the existing joint screen.

Budget: 36,000 actual + 180,000 shadow = 216,000 simulator calls. No new
reference labels. Raw traces, trees, residual arrays and checkpoints stay
remote; scalar JSON/logs only. Eligible Linux inputs live on node007; no machine
or GPU preference, no Windows, no belief/controller integration.

## Limitations

Conditional density adds capacity and query cost. One source seed and mild
known profiles; diagnostic mode recognition is not control-performance evidence.
