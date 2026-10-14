# v88 Mean-bias result and next step

Five local tests passed. Native smoke `t138447` completed in 53.26s; formal `t138450` completed on node007 CPU, exit 0, in 160.01s. All 48 direct baseline mean queries and 18 fold/control comparisons match the frozen sources. Smoke bias was reused. Only 29,487-byte smoke and 108,516-byte formal scalar results returned locally.

The correction changes accuracy from 86.0677% to 86.0352% and classification NLL from 1.2285603 to 1.2371147 (+0.696%). Three fold losses improve and three regress; four fold accuracies improve and two regress. Although 18/30 fold-mode losses improve, normalized-target prediction RMS worsens in 20/30 cells. All four earlier localized NLL regressions remain worse than raw. Normal and compound average accuracy decline; only slippery average NLL improves.

**Decision:** HOLD, retain v78. Do not advance this bias correction to the 65-path replay or tune its strength. Cross-seed average residuals do not provide a stable mean repair here.

**Next:** use the two saved v76 updating-normal reference traces, confirmed present on node007. Compare adding 1,100 sampled normal-prefix rows per seed (`0:3800`) with adding the same number of unused odd-index normal-suffix rows (`3801:6000:2`). Both add 2,200 input rows and 11,000 cached counterfactual labels; do not duplicate the normal/slippery paired prefixes. Keep v83 covariance, weight, forward fit and scaling fixed, change only the mean-fitting reference bank, and hold each source seed out in full. Reuse the same six validation populations before considering query replay. This matched-count comparison tests prefix coverage rather than merely adding more labels; no new task has been submitted yet.

## Limitations
These are previously inspected reference folds, not independent confirmation or RL return evidence. Inner errors use smaller fitting banks; covariance stayed at its pre-correction second moment. Additional reference labels in the next comparison must be reported separately from the original budget.
