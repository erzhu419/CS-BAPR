# v88 Nested mean-bias correction pilot

At each input, add to each frozen v78 local-linear mean the average target-minus-prediction error in its 64 reference neighbors. Each error comes from a mean fit excluding its entire source seed. For each outer validation fold, remove the outer seed before constructing any inner error bank or neighborhood. Coupled streams remain in the same seed. The correction strength is fixed at 1; no new tuning objective or mode-specific strengths are introduced.

Use the six v83 folds, 512 inputs per fold, unchanged five-mode labels and v83 candidate-specific mixed covariances (raw weight 0.24619454443573643). Keep the global forward model, scaling, ridge coefficient 0.064 and temporal rules fixed. The new offset is `old_offset - estimated_bias`; covariance determinants and components do not change.

Tests detect wrong correction sign, candidate/neighbor ordering, seed leakage, target/future dependence and unnecessary inner refits. Native smoke uses the first complete fold, checking eight direct baseline mean queries and all three v84 control summaries before scoring. A mismatch blocks formal dispatch. Formal evaluation reuses the smoke bias cache and covers all six folds; all error/bias arrays remain on node007. Only scalar JSON/logs return locally.

Compare classification NLL, accuracy, margins, confusion and normalized-target error RMS by fold and mode. Inspect normal and heavy controls as well as slippery/compound failures before deciding on any query replay. No evaluation traces, normal-null calibration, actor training, new seeds or manuscript changes are included.

## Limitations
Earlier weight selection and diagnostics used these folds, so this is not independent confirmation. Inner errors come from smaller fitting banks than outer prediction or deployment. For this isolated mean intervention, covariance remains the pre-correction second moment. Instantaneous reference classification is not recursive query accuracy or RL return evidence.
