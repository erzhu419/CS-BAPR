# v85 Classification-weight results

Task **t138428** completed on node007 CPU, exit 0, in **64.09s**. Four local tests passed; all three old classification losses reproduced v84. Only the **99,481-byte scalar JSON** returned: `results/general_ood_v85_classification_weight/pilot/classification_weight.json`.

Classification selection changes raw weight **0.2461945 -> 0.3235672**. Reference NLL improves **1.228560 -> 1.219822** (0.71%), but accuracy falls **86.07% -> 85.98%**. Only **2/6** fold losses improve; **16/30** fold/mode combinations improve and **14/30** regress. Mode-average high_grav/compound NLL worsens.

Four known regressions are mitigated, not removed: new/old/raw NLL is **4.1100/4.4975/3.5646** (135171 normal), **3.1098/3.3099/2.8758** (135171 slippery), **0.2045/0.2068/0.1839** (135050 slippery), **0.5626/0.6083/0.4936** (135092 heavy).

Decision: **HOLD, retain v78; do not advance this weight to a 65-path replay.** Stop this shared-weight objective branch. Next: diagnose failed query windows and matched successful controls, separating mean error along mode-separation directions from underestimated reference error in those directions. Actor, environment and paper tables are unchanged.

## Limitations
Weight selection and diagnostics reuse the same reference folds, not independent confirmation. Instantaneous classification losses do not establish recursive query accuracy or RL return improvements.
