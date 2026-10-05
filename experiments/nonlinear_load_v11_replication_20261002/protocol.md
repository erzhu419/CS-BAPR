# Nonlinear load v11: independent-training-seed confirmation

Freeze eight tasks before their results: training seeds 7/8 for `csbapr-centered-mlp-jacobian-only-no-target-shift`, `csbapr-centered-mlp-no-sindy-no-target-shift`, `bapr-no-target-shift` and `resac-no-target-shift`. The candidate keeps centered tanh MLP, no NAU, critic symbolic weight 0 and actor Jacobian weight 0.01. Reuse the exact validated v10 runtime and preset configurations; change only training seeds and experiment/output identifiers. No new loss, environment, weight or fitting adjustment.

Keep v9/v10's nonlinear actuator `5*(0.5*u+0.5*u^3)`, command-effort cost, six observations including current load, persistent Pareto/background disturbance and radius-0.2 reset. Train each policy at severity 1 for 500 episodes, 100,000 learner interactions and 99,001 updates. Only the candidate fits degree-one SINDy with its training seed: 10,000 identification and 1,000 validation interactions, threshold 0.001, unchanged feedback builder and learner RNG re-seeding after identification. Any builder failure is HOLD; do not adjust the intercept, threshold or data distribution to rescue a seed. Baselines retain their previously documented stabilizations and use no symbolic fitting.

Evaluate the last checkpoint on severities 1/2/5/10/20/50 and all paired paths 50000--50099. Every task includes the frozen known local-linear and inverse-actuator controls; the candidate also includes its own saved fitted feedback, without additional fitting. Save raw returns, reference gains, full configuration, runtime versions, final alignment and actual budgets in small JSONs.

Apply the existing comparison criterion to the new cohort alone: lower mean cost and 2/2 matched-training-seed wins at ID and 50x versus each of no-symbolic/BAPR/RE-SAC. Separately require positive learner-minus-own-fitted mean returns for both new training seeds at ID and 50x. Report every seed/severity and negative path gap. Seeds 5/6 informed candidate selection, so their results remain development evidence; a four-seed summary cannot override a failure on the seed7/8 confirmation cohort. This is independent training-seed confirmation on the existing evaluation paths, not held-out paths, new environments or general RL transfer.

The source is unchanged from the validated v10 snapshot. Reuse its candidate CUDA smoke and v9's validated baseline runs instead of repeating identical preflights. Verify the formal configurations against those frozen references before submission. Scheduler chooses compatible idle Linux/WSL resources without node preferences, allowlists or GPU pins; Windows dispatch is prohibited. Use pinned packages and full-state episode-boundary resume every 20 episodes. Checkpoints remain on workers outside result directories; automatically return only small evaluation JSONs. Preserve all previous negative gates and manuscript tables.

## Dispatch

At startup, all eight tasks were running and had nonzero optimizer updates. Worker-side checkpoint metadata matched the frozen full configuration, environment, evaluation protocol and runtime versions. Both candidate seeds built finite 2-by-6 fitted feedback gains. No training traceback was observed. Startup metadata is saved in `results/nonlinear_load_v11_replication/startup_observations.json`; no checkpoint was downloaded.

| Method | Seed 7 task and initial placement | Seed 8 task and initial placement |
| --- | --- | --- |
| Jacobian-only | t120854, jtl110gpu2 GPU0 | t120858, jtl110gpu2 GPU0 |
| No-symbolic | t120855, jtl110gpu2 GPU0 | t120859, local GPU0 |
| BAPR | t120856, jtl110gpu2 GPU0 | t120860, jtl311linux GPU0 |
| RE-SAC | t120857, jtl311linux GPU1 | t120861, jtl110gpu GPU0 |

Placement was selected by the scheduler without machine or GPU preferences. A transient staging transport failure recovered through normal retry. Initial placement is recorded for provenance, not pinned for resumption.

## Completion

All eight tasks finished with exit code 0. Final configurations, 100,000 learner interactions, 99,001 updates and 100 paired paths per severity match the freeze. Final fitted gains match startup metadata. The seed7/8 cohort passes all three learner-comparison gates and the added-value gate independently of development seeds 5/6. See `results/nonlinear_load_v11_replication/summary.json` and `docs/nonlinear_load_v11_replication_results.md`. The next step is frozen-policy evaluation on unused paired paths, without retraining or tuning.
