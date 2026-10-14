# v49: startup observation uncertainty

Retain v48 frozen policies, normal references, profiles, budgets1792, TimeLimit1536,
and the unchanged global residual score/log1p signal. No agent or actor update.

Normal calibration defines q=max(MSE(startup signal about ongoing signal mean)
- ongoing signal variance,0). Only at observable age0, add q to the df4 predictive
variance. Compare global-robust, noise-likelihood (history unchanged), and
noise-weighted-history (history precision=base_variance/(base_variance+q)).
All other observations have noise0 and precision1. Preserve physical-step hazard,
capped ages, history and nonzero startup contributions. No belief reset/masking.

Keep seven v48 trajectories per seed. Add normal -> heavy/slippery at the first
actual normal reset, determined from the preceding done, paired with the same
normal trajectory. Fixed-minus2 reset is at1536; relative reset timing is observed,
not assumed. Match complete prefixes, switch inputs and filter histories.

Retain first16/50/256 and last64 of the assessed first256, 50 per-step paired
penalty differences, startup same-past noise counterfactuals and physical endings.
Legacy global summaries, rewards, residuals and events must exactly match v48.

Preflight: 2 evaluation seeds/policy, 89088 interactions. Formal: existing
20 seeds/policy, 718848 interactions (24 reference and 360 physical test rollouts).
Three arms reuse each physical trace. Checkpoints stay on node007; scheduler
chooses idle GPU without preference. Scalar JSON/log-only transfers, no Windows.

The noise budget includes normal startup bias/spread. Fractional NIG history and
df4 variance addition are moment approximations, not exact heteroscedastic BOCD.
Keep failures; no preflight-driven tuning, new training seeds or RL feedback.
