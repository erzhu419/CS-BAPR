# v50: scalar departure survives; age is not OOD severity

Formal t135940 completed on node007 GPU0 in 457.04 s (7.6 min), with 432128
interactions: 24 normal reference and 200 test rollouts. Preflight t135939 took
84.68 s; four focused tests passed. All five trajectories, signals and both
filter source summaries reproduce v49 exactly. Only 7084328 bytes of scalar
curves/window-moment JSON returned, no trajectories/checkpoints.

At real reset switches, first50 log-signal differences are +2.01676/+2.91500
for fixed heavy/slippery and +.83683/+.52191 for relative heavy/slippery, each
20/20 positive. Noise-likelihood penalty differences remain -.10257/-.10283
and -.06174/-.00112, with 0/0/0/9 positive cases respectively. Positive signal
and negative penalty coexist in 71/80 pairs. Clock switches instead have both
contrasts positive in 80/80. All switch/stay first50 windows have zero endings;
all reset-switch/stay first256 windows also have zero endings.

Persistent fixed-policy OOD shows the distinction clearly: reset-switch last64
signal is 3.51478 heavy or 4.87316 slippery versus .12238 normal, while inferred
ages recover to 18.99/18.97 and penalty to about .150, below normal .15673.
Directional moments are available, but scalar compression has not erased the
tested departure contrast. This candidate's age readout describes recent changes,
not persistent deviation from normal dynamics.

## Limitations and next step

Two reused frozen seed1 policies; these auxiliary filters never control actions.
No RL improvement or mode-classification confirmation. Global-q95 exceedance
after a normal reset is 87.8% fixed and 29.8% relative over first50, so the raw
score is not ready to become a calibrated risk coefficient directly.

Next change the readout, not predictor capacity: retain change-point age for
change response, separately calibrate continuous normal-departure risk including
normal startup transients. Preserve observations and filter history. Then test
the resulting penalty mapping against fixed-minus2 with matched starts/budgets;
do not require every OOD event to shorten run length. No training-seed expansion
or paper-claim update in this round.
