# v84 Reference discrimination results

Task **t138426** completed on node007 CPU, exit 0, in **30.79s**. Four local tests passed. Six cached folds supplied 3,072 inputs / 76,800 candidate scores. All 48 direct mean checks and v83 loss comparisons passed. Only **199,602 bytes** of scalar JSON returned.

Result: `results/general_ood_v84_reference_discrimination/diagnostic/reference_discrimination.json`.

Raw / pure cross-seed / mixture accuracy: **83.25% / 86.31% / 86.07%**. Classification NLL: **2.2256 / 1.7586 / 1.2286**. The mixture improves every fold and mode average; this does not support attributing all query failures to the absolute-density objective.

Localized NLL regressions remain: seed135171 normal **3.5646 -> 4.4975**, slippery **2.8758 -> 3.3099**; seed135050 slippery **0.1839 -> 0.2068**; seed135092 heavy **0.4936 -> 0.6083**. The two v76 reference seeds remain hardest (**72.93% / 80.23%** mixture accuracy); seed135171 slippery-to-compound confusion is **22.46%**. Reference results do not reproduce v61's near-total query failures.

Decision: **HOLD, retain v78.** Next: a reference-only classification-NLL weight-selection pilot; inspect localized probability losses before another replay. If tradeoffs persist, investigate query/reference geometry. No actor or paper-table changes.

## Limitations
Post-selection reference diagnostics, not independent confirmation. Outer labels leave mean/covariance fits but contributed to weight selection. Instantaneous uniform-prior classification is not recursive query accuracy or RL return.
