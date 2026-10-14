# v56: reproduced threshold suppression

Task t136475 completed with exit0. The 13,477-byte scalar result reproduces all three cells, v55 age thresholds within1e-12, and the saved candidate control/change windows. Added cost3600 interactions/updates; no reference or gate changed.

At ages128/144/160/176/192, every pooled threshold exceeds the maximum of its post-reset normal calibration group. At age160 the initial q95 is0.576607 versus post-reset0.079385; at age176 these are0.642074 versus0.059258.

Failed gravity98723 last64 covers post-reset ages136--199. Its signal mean is0.121628 versus normal0.057429, but mean threshold is0.415481 and no step exceeds its age threshold. Both coefficients are zero. Before the reset, gravity exceeds the threshold at every step in the preceding64 window.

This localizes suppression to a normal reference mixing initial and later episode populations. v55 remains HOLD.

## Limitations
These are reused trajectories; learning progress and episode identity covary here. Generalization of a replacement and adaptive RL benefits remain untested.

## Next
v57 initial/post-reset age calibration, with a pooled control fitted on exactly the same new normal data. Each calibration runs2000 steps to cover ages0--999 in both populations. New paired test seeds, unchanged physics/model/scales/gates.
