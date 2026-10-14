# v150 frozen action-noise timing attribution

Status: complete. Two native smokes and four formal CPU tasks on node007
(t152176--t152179) validated, freely placed from node007/jtl110gpu2.
Results: `docs/general_ood_v150_noise_timing_results.md`.
Freeze all four v149
predeclared final700k SAC ZIPs; no650k selection, training, replay or updates.
Use the v149 frozen core and native CPU SB3 SAC2.7.0/Gymnasium/MuJoCo runtime.

Physical condition: baseline gravity1.0, held-out gravity2.6591479484724942 for
10 steps after100 or300 completed steps, then baseline without state reset.
Native rewards, motor bounds, healthy termination and1000-step cap unchanged.

Noise is sampled in pre-tanh action coordinates from the unchanged actor
mean/std; common absolute-step Gaussian draws are consumed even during quiet
windows. Compare scale1 throughout, scale0 throughout, and scale0 only during
the prefix,10-step pulse,100-step recovery, or pulse plus recovery. Quiet
windows return to native sampling afterwards; no fitted attenuation constants.
Clock boundaries are zero-based completed steps: prefix[0,onset), pulse
[onset,onset+10), recovery[onset+10,onset+110).

For every time-window program run an identical-clock no-physical-shock control.
Global native/quiet no-shock controls are shared between onsets, within the
same execution, not taken from old cross-host measurements. Verify identical
pre-shock state/return for each physical pair, and identical prefixes up to
the first noise intervention. Recovery-only also shares the full peak
trajectory with native sampling. Keep early/unexposed falls in every denominator.

Predeclared fresh reset seeds155000--155004 and action-noise seeds139000--139002.
Sampled programs have15 reset/noise repeats per condition; global quiet has
five deterministic resets, without three duplicate noise repeats. There are
300 episodes per policy,1200 total,640 physical-prefix pairs and960
noise-boundary pairs. These remain two paired training-source seeds.

Report ID controls, final healthy survival, return/length,100-step recovery
and delayed falls separately per model/window/onset. Examine shock-specific
benefit relative to the matched no-shock control; do not pick a winning window
as a deployable result. A reproducible local effect motivates a later causal
observation-based noise controller with independent evaluation. If quieting
does not help, prioritize mean-policy/recovery-state support instead.

## Limits

The evaluator uses the true shock clock. This is an oracle diagnostic, not
CS-BAPR, a learned detector, or deployment evidence. Any deterministic/sample
comparison retains its actual5/15 repeat counts. Two native smokes precede
four freely scheduled CPU jobs on available Linux nodes; no Windows, machine
preferences or hard pins. ZIPs stay server-side; pull only small JSON/logs.
