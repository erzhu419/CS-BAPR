# v73: temporal inference with a frozen v72 density

Reuse the saved v72 conditional-density bank unchanged. Compare its original
16-step mean log-density readout with a recursive finite-mode posterior:
`prior = 0.98 * posterior + 0.02 / 5`, followed by log-space likelihood update.
Start uniformly; preserve mode history across episode resets. This is a new
mode filter, not the existing scalar-surprise run-length tracker. The 0.02
uniform-redraw hazard is fixed before tests, with actual off-diagonal change
probability 0.016 per step; no parameter search or evaluation-label fitting.

Recursive departure uses softplus of the log ratio of mean changed-mode
posterior mass to normal posterior mass. It is a posterior contrast, not a
Bayes factor. Recalibrate only its phase/age curves on the same 12,000 v69
normal-null transitions. Keep quantile, age neighborhoods and risk smoothing;
reuse the baseline curves verbatim. Both readouts share the exact density
values, including covariance volume and physical-coordinate normalization.

Freeze references before seeds 135140/135141. Each seed has normal, heavy,
slippery, high_grav and compound branches: 6,000 steps, switch 3,800, native
TimeLimit 1,000, stochastic frozen source actor and causal context. Retain
paired prefixes and exact true-mode replay checks. Budget: 60,000 actual plus
300,000 shadow = 360,000 simulator calls; zero optimizer updates/new labels.
Replay v72/v71/v70/v62/v61 separately as development data.

The existing joint screen requires normal-window mean risk <=0.1 and positive
first50/last64 paired risk contrasts AND true-mode margins in every changed
cell. Window margins remain mean log-density differences; recursive margins
are log-posterior differences. Only their signs are gated; numerical magnitudes
are not comparable. Raw instantaneous margins diagnose density coverage.

CPU evaluation on Linux nodes holding the server-only inputs, without node/GPU
preference or Windows. Only scalar JSON/logs return locally. No actor feedback.

## Limitations

One trained source policy, mild known modes, privileged reference labels and
approximate likelihoods. Mode recognition is not an adaptive-control result.
