# v87 Shared conditional covariance pilot

Reuse the six frozen v83 whole-seed folds (512 inputs each) and their five local-linear conditional means. At each input, set every candidate covariance to the equal arithmetic mean of the five v83 mixed covariances. Retain the density-selected raw weight 0.24619454443573643, all covariance components, 64-neighbor geometry, ridge coefficient 0.064, scaling and global forward fit. There is one new condition, `shared_mix`, with no parameter selection.

This is a proper conditional Gaussian for each candidate. Candidate log determinants cancel because the covariances are identical; the shared precision metric also changes. This is not the v86 oracle-label common-true covariance and does not discard determinants while retaining different precisions. Neither the observed next state nor its true mode enters covariance pooling.

Before interpretation, reproduce each v84 fold's three old controls (raw, pure cross-seed, selected mixture), including per-mode losses, margins and confusion fractions. A mismatch stops interpretation until the cache/scoring correspondence is corrected. Formula tests check the pooling axis, unchanged means and normalized Gaussian scores.

Report classification NLL, accuracy, signed margins and confusion by fold and mode. Inspect the v84 localized regressions and preservation of heavy-mode success, not just the pooled average. Decide from these results whether a development-path replay is informative. This pilot loads no query traces or normal-null data, adds no environment steps, actor feedback or paper-table changes. Use node007's existing native caches; return only scalar JSON/logs.

## Limitations
The same reference folds were used by earlier weight selection and diagnostics; this is not independent confirmation. Covariance tying cannot in itself repair a wrong mean direction. Reference classification is not recursive path accuracy or RL return evidence.
