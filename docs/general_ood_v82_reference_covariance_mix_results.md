# v82 Reference-selected covariance mixture results

Native task t138070 completed on node007 CPU, exit 0, in 88.24 seconds. Five local tests passed. Server-side reconstruction matched all 23,000 reference rows and all five mode-residual arrays. The native check preserved v78 means/errors on 228 existing-path transitions and reproduced both v78/v81 density endpoints. Only the 18,820-byte scalar JSON was synchronized locally.

Six whole-reference-seed holdouts, each with 512 evenly spaced queries and all five known-mode labels, produced 3,072 queries / 15,360 labels. The shared raw-covariance weight was exactly **1.0**; the bounded interior solution was 0.999994728. Equal-seed/equal-mode validation NLL was **-21.1264** at the raw endpoint versus **-5.0977** at the pure LOO endpoint (lower is better; weight-independent constants omitted).

LOO was better on the source-normal seed 135050 and normal-state seed 135090 folds. Raw covariance was better on high-gravity/compound seeds 135091/135092 and both updating normal/slippery reference seeds 135170/135171. The latter two folds had raw/LOO NLLs 1.9311/35.7978 and -21.0160/18.6539. Local LOO error covariance does not generalize uniformly across reference trajectories.

Decision: retain the v78 raw-covariance endpoint; no new readout is supported by this selector. As predeclared, no duplicate 65-path replay was dispatched. This does not resolve v78's existing HOLD or establish improved mode identification.

Next: keep the mean fixed and calibrate uncertainty from whole-reference-seed-held-out prediction errors, not within-neighborhood LOO errors. Keep error-bank fitting and mixture-weight validation separate, with no held-out labels entering the validation fold's covariance calibration. Start on references before another full replay; no fresh seeds or actor feedback yet.

## Limitations
This is known-mode reference validation plus native equivalence, not new RL return or general OOD evidence. The two native paths are existing development traces. No new full replay, independent control replication or paper-table update was performed.
