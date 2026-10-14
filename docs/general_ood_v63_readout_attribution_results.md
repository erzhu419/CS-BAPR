# v63 replay attribution results

t136552 completed on node007 CPU, exit0, in26 seconds. All17 saved-trajectory
replays exactly match their original first50/tail risks and signals (error0).
The v62 summary is unchanged. Added interactions/updates0; only the696kB JSON
was synchronized. Four focused frozen-runtime tests passed.

Early gravity failure, seed134020: level signal0.342247 exceeds paired normal
0.194769, but recalibration raises the mean threshold from0.376277 to0.864350.
Joint signal is0.344081 against threshold0.872347; even its largest margin
is-0.218078. All50 samples are below threshold, so neither smoothing nor a
different episode phase explains the zero instantaneous risk here.

Tail gravity failure, seed134021: joint signal0.134790 remains below the
mean threshold0.310868, with largest margin-0.111331. All64 samples fall below
all three thresholds. Innovation never dominates in this window, although
paired signed-mean displacement energy is0.746435 in the frozen metric.
Both failed windows have0% input-energy exceedance above normal-fit95.

On reused v61 trajectories, first50 positive counts are1/4 for frozen v60,
1/4 for recalibrated level and2/4 for joint innovation; tails stay4/4. Innovation
repairs the slippery contrast (+0.048124), not gravity (-0.152487) or compound
(-0.094385). The normal joint first50 pulse is still0.357889.

Next: screen an observed phase/age-conditioned signed-vector center and
covariance on the existing separate normal-fit/calibration traces. Preserve
all11 coordinates, causal histories, normal bounds and early/tail contrasts.
Retrospective screening precedes new continuation seeds and closed-loop work.

## Limitations

Replays are not new independent samples. Matched vector displacement and
scalar input energy do not prove online identifiability or full input support.
No threshold was lowered, no reference refitted, and no RL return claim added.
