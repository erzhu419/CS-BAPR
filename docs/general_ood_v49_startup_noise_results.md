# v49: startup likelihood repair, not OOD mode identification

Formal task t135937 completed on node007 GPU0 in 809.79 s (13.5 min): 718848
interactions, 24 normal reference and 360 paired test rollouts. Seven focused
tests passed. Legacy v48 trajectories, scores, signals and control beliefs match
exactly; actual-reset switch prefixes and assessment windows match. Only
17982422 bytes of formal scalar JSON returned, no trajectories/checkpoints.

Both noise arms remove all 87 legacy fixed-policy episode-restart majority-reset
events, including normal cases 20/20 -> 0/20, without moving major events later.
The 120 nonstartup fixed-policy events remain. Relative-policy event counts remain
45, with no restart-majority events in any arm. Same-past startup counterfactuals
also remove the original restart triggers.

Legacy changed-pair first50 penalty contrasts remain positive in 159/160 cases.
Relative heavy -> slippery is +.07394, 19/20 for noise-likelihood; its seed90019
negative case remains. Weighted history adds no consistent benefit: normal fixed
restart first50 penalty is .34722 for noise-likelihood versus .35927 weighted
(global .37279); relative values are .33652 versus .33445 (global .33796).

Real changes at reset remain unresolved. Noise-likelihood first50 changed-minus-
normal penalty is -.10257 heavy and -.10283 slippery for fixed (0/20 positive
each); relative is -.06174 heavy (0/20) and -.00112 slippery (9/20). All four
reset-change/stay first256 windows have zero physical endings. The original
control already has negative mean contrasts; noise also removes its 40 fixed
major triggers at genuine reset switches. Quiet reset counts do not prove mode
identification, and a signed penalty contrast is not an RL return comparison.

## Limitations and next step

Two reused frozen seed1 policies, few normal starts (8/16), approximate noise
moments and no RL feedback. Keep noise-likelihood as the simpler diagnostic
candidate; do not carry weighted history forward or expand training seeds yet.
Next separate change-point age from normal-departure evidence: inspect signed
dynamics residuals and normal-calibrated departure over paired first16/50 reset
windows, retaining the same references, profiles and physical traces. Determine
whether the scalar compression or the run-length readout loses the distinction
before changing the belief-to-penalty mapping. Paper claims remain unchanged.
