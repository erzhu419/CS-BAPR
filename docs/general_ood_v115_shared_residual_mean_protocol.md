# v115: matched-reference shared residual means

Compare a shared MLP residual correction with the existing local-linear mean.
The MLP has two 128-unit SiLU layers and 55 outputs: five modes, eleven normalized
delta-state residuals each. Add its outputs to the frozen global forward model.
Input/target scales stay frozen. Fit equal-source-seed, equal-mode MSE using
AdamW, learning rate 0.001, weight decay 0.0001, batch size 512, seed 115 and
100 fixed epochs. Initialize the final layer to zero. No early stopping, tuning
or model selection against query windows.

First train on the eight incumbent reference sources / 29,600 inputs, and compare
both means on source135230/135231 / 4,400 paired inputs. Both full sources are
excluded before training either mean. Report per-source and per-mode prediction
MSE/RMS, paired nearest-mode accuracy, and errors exceeding simulator mode gaps.
The global forward model and scales are inherited rather than cross-refitted.

Then train the MLP on all ten sources / 34,000 inputs. Diagnose the same 76 query
windows / 4,248 transitions, retaining old failures, normal and frozen controls.
Recompute local-linear scores and match v114. Compare actual true-mode errors,
nearest-mean recognition and the frozen v113 formal classifier. The latter is
an input intervention, not classifier recalibration. Simulator outcomes are used
only in error/mode-gap diagnostics, never query prediction. No temporal replay.

Native smoke uses two epochs and 32 reference inputs per source, with the seven
v114 smoke windows. It tests loading, scoring, units and output but cannot select
a candidate. Formal runs use the full fixed budget. CPU 2, RAM 4 GB, GPU 1 GB;
node007 holds the native arrays. No Windows nodes or soft placement preferences.
Only scalar JSON returns locally; model states remain remote.

## Next decision

Review holdout errors and recognition together with all control cohorts before
considering full-history belief replay. Mean improvement alone does not establish
actor or return gains. Previously used source holdouts and query windows are
development evidence. No seed expansion or actor adoption in v115.
