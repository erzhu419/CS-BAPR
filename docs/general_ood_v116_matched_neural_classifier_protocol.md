# v116: classifier matched to frozen neural means

Freeze v115's formal source-holdout and final mean models. Replace only the
quadratic classifier/scaler, retaining v113's 110 residual/squared-residual
features, equal-source/equal-mode weights and logistic settings (C=1,
L-BFGS, tolerance 0.0001, maximum 1,000 iterations). No hyperparameter search.

For source-holdout validation, make eight neural prediction-error folds on the
29,600 incumbent inputs. Each fold excludes its entire source before mean
training; sources135230/135231 are absent from every fold, classifier and scaler.
Fit a classifier using these out-of-source features, then score the 4,400 held-out
inputs with v115's frozen eight-source mean. Controls use the original v113
eight-source classifier and nearest mean on the same neural predictions.

For the final classifier, make ten analogous folds over all 34,000 reference
inputs and fit on 170,000 paired labels. Every crossfit mean uses v115's unchanged
architecture, 100 epochs and training settings. Query inference uses the saved
v115 final model, never a crossfit or smoke mean. Reproduce v115 original-classifier
and nearest-mean scores and true-mode RMS on all 76 windows / 4,248 transitions.
Compare matched classifier, original classifier and nearest mean on all cohorts.

Native smoke uses two-epoch crossfit means and 32 error inputs per source, plus
the seven v114 smoke windows. Its query means are nevertheless the full formal
v115 models. All 18 folds and both classifier fits must complete before formal
dispatch. Smoke cannot select a candidate. Cache fold error arrays remotely so
a scheduler retry can resume completed folds without retraining them.

CPU 2, RAM 4 GB, GPU 1 GB. Native arrays reside on Linux node007; no Windows
placement or soft node/GPU preference. Only scalar JSON returns locally.

## Interpretation

Previously used source holdouts and query windows are development evidence.
The frozen global forward model and input/target scales are not cross-refitted.
Classifier probabilities are not normal-null calibrated temporal evidence.
Judge normal controls and remaining failures before considering full-history
belief replay. No seed expansion, actor adoption or manuscript change in v116.
