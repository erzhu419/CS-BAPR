# v50: normal departure versus change-point age

Retain v49 frozen seed1 policies, normal-only fit/calibration seeds, dynamics
predictor, startup-noise rule, profiles, TimeLimit1536 and budget1792. Report
global-robust and noise-likelihood. No optimizer, replay or actor feedback.

Each evaluation seed has five physical trajectories: normal, heavy/slippery
from1024, and heavy/slippery from the first actual normal reset. The latter
switch is determined by the preceding done, not assumed to be at1536. Compare
each switch with the same normal trajectory at its own observed switch step.
Complete physical/filter prefixes and switch inputs must match. Source summaries
for all five trajectories and both filter arms must exactly reproduce v49.

Use unchanged global normal residual centering/scaling. Record first16/50/256
and last64 of the assessed256: mean log signal, score/normal-q95, exceedance
fraction, nonstartup mean score, all coordinate signed means and squared means,
energy of the signed mean, belief statistics and physical endings. Keep first50
scalar signal/score/penalty curves; no per-transition residual vectors or states.
Coordinate indices follow the stored Hopper observation order; no mode-label fit,
coordinate selection, changed hazard or threshold tuning.

Preflight 2 evaluation seeds/policy: 60416 interactions. Formal existing20/policy:
432128 interactions (24 references, 200 tests). Checkpoints stay on node007;
scheduler selects an idle GPU without preference, no Windows nodes. Transfer only
compact scalar/vector-moment JSON and logs, no trajectories/checkpoints.

Signed moments are diagnostics, not a latent-mode posterior. Global calibration
is not startup/state-conditioned; normal-q95 is not a guaranteed error rate.
Retain failures and decide whether scalar compression or belief readout merits
the next change. No RL seed expansion or manuscript update in this round.
