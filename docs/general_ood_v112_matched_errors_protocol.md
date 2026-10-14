# v112: expanded-mean-matched prediction errors

v111 found that mean accuracy alone did not repair the failed likelihood rankings.
This pilot changes only the held-out error second moment used for uncertainty.

- Means: unchanged v106 standard extension, 34,000 rows from 10 reference seeds.
- Raw covariance: unchanged v98 bank, 27,400 rows; raw weight 0.24619454443573643.
- Error support: unchanged v98 29,600 inputs from eight seeds. The 4,400 added
  mean inputs do not become error-support inputs in this isolated intervention.
- Error fitting: for each support source seed, exclude ALL its mean rows and
  predict its support rows using the expanded mean. Refit all five mode errors.
- Covariance: same 64-neighbor OAS second moment and relative ridge. Preserve the
  old mean center and eight old members for the disagreement moment. Trace-match
  rescaling uses the new base covariance plus that unchanged moment.
- Controls: old v98 and v106 means with incumbent covariance; candidate expanded
  means with matched errors. Simulator-only correct centers with each covariance
  separate center error from covariance effects; they are not deployable arms.

The formal population has 76 windows / 4,248 transitions: v111's 14 current
failure, frozen-pair and normal windows, plus v105's 62 earlier failure/control
windows. All three covariance variants are retained. Report window-level
instantaneous accuracy, true-vs-best-other margin and energy/volume decomposition.
Compare unchanged old controls to archived attribution and fresh old/new
  likelihood slices to their native caches. The two historical v62 windows lack
  counterfactual observations: include all three real arms there, but no oracle
  arms. Each summary arm records its actual number of covered windows.

Native smoke checks two held-out rows per old reference seed and six windows /
328 transitions. Its covariance intervention is the identity using incumbent
errors; partial errors are never passed off as a fitted candidate.

No environment collection, optimizer update, coefficient tuning, normal-null
recalibration, temporal replay or actor integration. Results are development
diagnostics. A promising pilot still needs complete-history replay and normal
controls; a failed pilot does not justify more seeds or adoption.

Run beside the resident Linux node007 input artifacts. No soft node/GPU
preference. Synchronize scalar JSON only; native error arrays remain remote.
