# v73: faster recognition, with a tail-stability tradeoff

t136986: exit 0, 19m52s; 60,000 exact replays, zero optimizer updates/new labels.
Only scalar results were pulled; the fresh joint screen remains HOLD.

Fresh early accuracy rose 66.75% -> 86.00%; positive mode-margin pairs 4/8 ->
7/8. Tail accuracy fell 95.90% -> 92.38%. Compound tails fell 100%/100% ->
93.75%/84.375%; one slippery early margin remained negative. Largest recursive
normal-window risk was 0.02065 (<0.1). Margin units differ; compare signs only.

A zero-step density decomposition found slippery raw early accuracy 54%/50%,
negative mean margins and compound as best competitor in 92% of transitions.
Mean-model error was about three times the nearest mode observation gap.
Strong incorrect compound tail evidence also remained (minimum -136.57).

Next: freeze forward coefficients/scales, temporal readouts and hazard; compare
equal-count normal-state versus slippery-state density coverage extensions.
Recalibrate using the original normal-null streams/rules, then fresh all-mode
tests. No controller integration or v74 submission yet.

## Limitations

One source policy and known mild profiles; historical updating failures remain.
These are mode-recognition diagnostics, not adaptive-control evidence.
