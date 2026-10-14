# v51: normal-age departure readout

Formal t135942 finished on node007 GPU0 in 525.53 s (8.8 min), with 503808
interactions, zero optimizer updates and zero replay insertions. All v50 source
records, references and loaded-policy metadata match exactly. Only 8230687 bytes
of scalar JSON returned; four focused tests passed after preflight. Full results:
`results/general_ood_v51_departure_penalty/summary.json`.

Age-departure separates changed/stay first50 windows in 160/160 pairs, and
last64 windows in 160/160. Reset heavy/slippery first50 differences are
+.31606/+.47159 for fixed-minus2 and +.13322/+.07753 for relative-adaptive;
last64 differences are +.95263/+.98550 and +.64650/+.62211. Unlike the old
run-length coefficient, sustained departure does not disappear after adaptation.

Across 20 new normal trajectories per policy, global-to-age whole-run mean
coefficients drop .04392 -> .00272 (fixed) and .03734 -> .00464 (relative).
Restart first50 means drop .57042 -> .01262 and .26606 -> .02833. There are
20/40 complete restart windows; their largest age-departure means are .05381/
.05235. Initial first50 means are .01729/.02039, with maxima .08118/.05366.
The candidate reduces normal startup penalties while retaining OOD separation.

## Limitations and next step

Relative reset-heavy first16 is positive in only 13/20 pairs (mean +.00075,
minimum -.05932); other seven scenarios are 20/20. Normal whole-run peak
coefficients still reach .65159/.50513. This is a bounded readout, not a calibrated
mode probability. Two frozen seed1 policies and reused OOD trajectories establish
mechanism diagnostics, not RL gains or independent confirmation.

A matched-start training pilot is justified. Use the same fixed-minus2 checkpoint
for all arms, shared replay refill and stochastic-action normal fit/calibration
data: actual training samples actions, whereas these probes use deterministic
actions. Compare age-departure with fixed-minus2, fixed-minus2.3 and the old age
mapping under identical added budgets and evaluation seeds. Checkpoints omit
replay, so call this warm continuation, not exact resume. No training or manuscript
claim changes were made in this round.
