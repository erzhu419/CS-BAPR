# v46: Crossed calibration and predictive tails

Freeze the v40 seed1 Hopper policies, v43 independent normal reference splits,
quadratic-state-action predictor, test seeds, physical profiles, 1536-step
TimeLimit and normal-to-target switch at 1024. No actor/critic training, replay
insertion, context changes or new training seeds. Formal budget remains 24
normal reference plus 200 matched test rollouts: 380,928 interactions.

Four arms share every physical trajectory, H=20, hazard=.02, fresh variance=1,
uniform initialization, 19-sample history and no episode-end belief reset:

- `episode-original`: episode-centered calibration, v44 AdaptiveStudent.
- `episode-robust`: same calibration, variance-matched df=4 predictive tails.
- `global-original`: existing global-centered calibration, v44 AdaptiveStudent.
- `global-robust`: same global calibration, variance-matched df=4 tails.

All inputs are `log1p(score / calibration95)`. Conditional learned means and
predictive variances are unchanged when replacing tails at identical history;
subsequent posterior paths can differ. Lambda and hypothetical beta retain
`E[exp(-.1 * age)]` and `-2-lambda`; neither controls actions. No parameter
selection from preflight or OOD results. Retain all arms/profile outcomes.

Add whole1536, prefix1024 and post512 measurements to all v44 windows.
Export every posterior-majority-reset event index plus scalar age, signal,
boundary and ending indicators, per-window transition counts, mean/peak/end
posterior metrics and matched sham lambda contrasts. This prevents a quiet
selected window from concealing normal false triggers elsewhere. Only compact
JSON/logs are synced, no full states/actions or checkpoints.

Require unchanged v44 reference/model/config/runtime, residuals, rewards/endings
and episode-original belief statistics. Global-original must also reproduce
v45 on overlapping cases. Tests cover causal prefixes, original control,
whole-trace event accounting and source drift rejection.

## Interpretation and Next Step

Compare full normal false triggers alongside soft lambda/age shifts, early
paired response, and late recovery; reset probability >.5 is descriptive, not
a tuned performance gate. A calmer candidate that loses real change response
is a tradeoff, not success. Then test repeated mode changes before matched-start,
matched-budget RL. Reused development replay is not independent confirmation.
