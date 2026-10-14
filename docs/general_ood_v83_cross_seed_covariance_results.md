# v83 Cross-seed covariance results

Eight local tests passed. Native task t138096 completed on node007 CPU, exit 0, in 233.79 seconds. The 228-transition check preserved v78 means/errors and raw densities, matched the pure cross-seed error second moments and verified density/readout agreement. Only the 9,254-byte scalar JSON was synchronized locally; the full 23,000-row error library and nested fold caches remain on the server.

Six nested whole-seed holdouts selected raw weight **0.2461945444**, cross-seed-error weight **0.7538054556**. Equal-seed/equal-mode selection NLL was **-25.4178**, versus **-23.6162** for pure cross-seed errors and **-21.1264** for raw covariance. The mixture beat raw covariance in 5/6 folds and all five mode-wise averages; source-normal seed135050 slightly regressed (-25.5236 versus -25.9987).

Formal task **t138124** completed on node007 CPU, exit 0, in **5,477.86 seconds (1h 31m 18s)**. Both conditions completed all 65 paths and the same 12,000-transition normal-null calibration per condition. Scalar synchronization succeeded; the 4,124,826-byte JSON contains window statistics, not transition arrays. All six reused controls match v78/v81 exactly, and all 260 new-condition window RMS comparisons preserve v78 means. No actor, environment or paper table changed.

Result: `results/general_ood_v83_cross_seed_covariance/cross-seed-covariance/cross_seed_covariance.json`.

## Development comparison
- Negative recursive true-mode-margin windows: v78 **14/96**, v81 **18/96**, pure cross-seed **11/96**, selected mixture **12/96**. Both new conditions resolve five v78 failures; pure introduces two, mixture three. Five of eight replay groups pass the existing screen, versus four for v78. v76, v62 and v61 still fail.
- v76 early/tail accuracy: v78 **56.50%/85.55%**, pure **55.50%/89.26%**, mixture **56.50%/88.48%**. Tail gains do not repair early failure. Seed135180 high_grav early accuracy regresses from **64%** to **12%/28%** (pure/mixture).
- v62 early/tail: v78 **73.50%/62.89%**, pure **87.75%/65.23%**, mixture **86.50%/62.89%**. Mixture introduces a negative high_grav tail margin on seed134020.
- v61 heavy tail improves from **15.62% to 100%** under both conditions, but slippery tail falls from **87.50% to 3.12%/1.56%**. High_grav tail remains **0%**. Consequently, group-average tail accuracy barely changes: **49.61% -> 50.78%/50.39%**.
- Normal mean risk remains below 0.1: maximum **0.02733** (pure), **0.00761** (mixture), versus **0.00796** for v78. All 96 changed-window paired risk contrasts remain positive. The remaining rejection is mode evidence, not this normal-risk screen.

Decision: **HOLD; retain v78 as the working baseline, keep v83 as a development candidate.** Do not replace the reference-selected mixture with the pure endpoint based on these replay outcomes. Whole-seed error calibration is more promising than v81 LOO calibration, but neither new condition establishes reliable mode identification.

The failures are heterogeneous. v76 early slippery/compound prediction RMS (0.4393/0.4521) exceeds the measured nearest-mode observation gap (0.0962/0.1175) on every transition in those windows. In contrast, v76 seed135181 heavy tail has RMS 0.0968 versus gap 0.1493, yet high_grav is the best competing mode on 93.75% of transitions and the mixture's mean energy advantage is -7.119. In v61 slippery tail, the energy advantage remains positive (+1.4995), but the covariance-volume term (-2.1321) reverses the instantaneous mean margin to -0.6326. Neither a single mean-error explanation nor covariance widening explains every failure.

Next: inspect five-way relative likelihoods on the existing six whole-seed reference holdouts, especially slippery versus compound/normal and heavy versus high_grav. Absolute true-mode Gaussian NLL does not directly measure discrimination. If the holdouts reproduce these ranking failures, change the reference-only selection objective; if they do not, target reference-to-query geometry rather than tune to replay labels. No new seeds or actor feedback yet.

## Limitations
The reported NLLs are weight-selection losses. All 65 paths are development replays from one source training seed, not independent confirmation, new RL returns or general OOD evidence. Cross-seed errors use smaller fitting banks than deployment and can overstate uncertainty. The six controls are reused statistics; no fresh seeds have been run.
