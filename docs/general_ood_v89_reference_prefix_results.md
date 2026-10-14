# v89 Matched coverage result and next step

Six local tests passed. Smoke `t138662` completed in 163.17s; formal `t138664` completed on node007 CPU, exit 0, in 129.34s. Source rows/all five labels, 48 baseline mean queries and 18 control comparisons match. Added rows are included in whole-seed exclusion, with equal training counts. Only scalar results (39,092/148,760 bytes) returned locally.

Prefix accuracy rises from 86.0677% to 86.7969%, improving in all six folds. Classification NLL falls from 1.2285603 to 1.1487982 (-6.49%); four folds improve and two slightly regress. The equal-count suffix control reaches 86.1849% and NLL 1.2371267. Prefix beats suffix accuracy in five folds and NLL in four; its aggregate NLL is 7.14% lower. Prefix prediction RMS improves in 23/30 fold-mode cells.

All five mode-average prefix losses improve, but slippery accuracy declines from 83.8867% to 83.5286%. In seed135050/slippery, accuracy falls from 91.7969% to 88.8672% and NLL rises from 0.2067655 to 0.2621587. All four earlier localized NLL regressions remain worse than raw. The aggregate gain is therefore not a uniformly better classifier.

**Decision:** keep v78 as the working baseline; admit both fixed prefix and equal-count suffix conditions to a matched development replay next. Do not adopt the new readout, expand seeds or enable actor feedback yet.

**Next:** replay the same 65 saved paths, with means fitted on the complete added reference banks but v83 covariance/weight unchanged. Calibrate each readout on the same 12,000 normal-null transitions using the unchanged temporal rules. Retain both additions and existing controls; inspect v76 early failures, v61/v62 tails, slippery regressions and normal risk before advancing. No replay task has been submitted in this round.

## Limitations
Previously inspected reference folds are not independent confirmation or RL returns. Each addition uses 2,200 more inputs/11,000 cached labels; the suffix control matches that budget. Covariance support remains fixed. Prefix/suffix differ in their state/action distributions, not only clock time.
