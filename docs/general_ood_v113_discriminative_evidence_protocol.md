# v113: paired-counterfactual discriminative evidence

Keep the v106 expanded mean, forward coefficients and scaling. For each reference
state/action, use all five saved counterfactual targets with equal mode weight.
Features concatenate the 11-coordinate residual against each of five predicted
means and its coordinate squares (110 dimensions). Fit weighted StandardScaler
and multinomial L2 logistic regression: C=1, lbfgs, tolerance 1e-4, max_iter=1000.
Equal source-seed weights prevent the largest source from dominating. No search.

Reference features exclude every row of the target's source seed from the local
mean fit. Existing v112 errors cover 29,600 rows; fit the remaining 4,400 rows
from seeds 135230/135231. Final classifier: 34,000 inputs / 170,000 mode labels.

Before final fitting, hold out both 135230/135231 entirely: fit the validation
classifier/scaler on old eight-source references with their old whole-source-
held-out means, and score the two held-out sources using only the old mean bank.
Neither validation source can influence any training feature or preprocessing.
This is one fixed two-source validation split, not ten-fold classifier validation.
The global forward model and target scale remain frozen throughout.

Compare final classifier and fixed nearest-normalized-mean control on the same
76 windows / 4,248 transitions as v112. Copy scalar v98/v106/v112 Gaussian
controls, explicitly marking reuse; compare new true-mode prediction RMS to
all 228 saved condition/window values to verify the input and mean contract.
Use only observed states, executed actions and next states for query scoring.
No query mode label, oracle observation, future transition or seed is a feature.
Log probabilities are discriminative scores, not physical Gaussian densities.

Smoke uses at most 32 inputs per reference source, 32 per validation source and
six windows / 328 transitions; its partially trained classifier is not a candidate
selection result. Formal fitting has two classifiers and no RL optimizer updates.
No collection, actor integration, temporal replay, threshold calibration or seeds.
Use resident Linux node007 artifacts; sync scalar JSON only, not reference arrays.
