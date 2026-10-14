# v61 early-response localization

Reuse132020 deliberately: five matched6000-step updating cells at beta=-2,
normal/heavy/slippery/high_grav/compound, with the same3800 switch and frozen v60
long-normal refit. Total30000 interactions/updates. Compare first50 and last64
coefficient/signal means against completed v60 within1e-8 before interpreting
these reruns as reproductions. This is not new independent evidence.

Report pre64, post0--10/10--20/20--30/30--50/50--100/100--200, first50 and
last64 windows. Separate scalar score, log signal, threshold, instantaneous
risk and16-step smoothed risk. Preserve both readout histories across resets.

Use the common pre-switch last64 mean signed-residual moment as a diagnostic
anchor. In the fixed precision metric, level score equals anchor score plus
the cross term plus squared displacement. Report matched changes in all three
terms and all11 observation-order coordinates. The known boundary/anchor is
offline annotation only; it does not alter actions, beta, references or gates.

Checkpoint native/replay/RNG progress every500 steps. Keep every completed raw
trace under the server checkpoint directory; return only scalar JSON/logs.
