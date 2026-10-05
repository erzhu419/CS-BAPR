# Nonlinear Load v30: Sustained Load Qualified

The sustained-load candidate and admission criteria are frozen in the [protocol](nonlinear_load_v30_environment_protocol.md). Forty-two focused/regression cases passed in7.51 seconds, including unchanged legacy physics/reference controls, latent-draw pairing, action-independent load streams, window accounting and rejection of inconsistent evidence. Launcher syntax and scoped scheduler doctor passed.

CPU preflight t132251 exited0 and its41,645-byte JSON passed validation. Formal tasks t132277/t132278/t132279 exited0 in32.43/32.42/117.54 seconds on scheduler-selected local WSL/local WSL/node007 CPUs. All6,000 controller episodes/1,200,000 transitions passed frozen validation, including100 paired paths per block, common action-independent loads and four-window physical-cost accounting. Formal JSON totals3,078,645 bytes; two reports were produced locally and only the radius1 report returned from the server. There was no fitting, RL training or checkpoint access.

All three radii pass the prespecified environment criteria. At50x, the feasible force-cancel reference has:

| Radius | Cost After Step50 | Maximum State Norm | Near-Saturated Components |
| --- | ---: | ---: | ---: |
| 0.2 | 75.10% | 0.286 | 0.0075% |
| 0.5 | 74.00% | 0.711 | 0.2025% |
| 1.0 | 69.84% | 1.423 | 0.8525% |

The process creates continuing physical cost rather than only initial regulation. Its burden is mainly command effort:93.27--99.74% of reference total cost. Correct current-load compensation reduces late state cost to4.07e-7/2.62e-6/1.33e-5 at the three radii, versus24.14/24.17/24.28 without load observation. Thus repeated loads matter, but the observed-load plant remains easy for a known-model controller. Force cancellation is not optimal: the legacy known-inverse controller costs less in all three OOD blocks.

## Next Step

Propose a new two-seed learner pilot (seeds21/22), full symbolic Jacobian constraint versus constraint-off, with matched shared prefixes,500 episodes per arm and last-checkpoint evaluation. Retain v25 early-critic/uniform-actor sampling, automatic temperature/floor, risk rules and ID-only identification/training. Freeze all other settings and the original three-radius ID/50x physical-cost/own-fitted criteria on this new cohort before dispatch. This tests whether the symbolic constraint adds value under sustained disturbance handling. No sampling/weight sweep or BAPR/RE-SAC expansion yet; no learner jobs were submitted in v30. Historical HOLD and manuscript remain unchanged.

The user approved this step on 2026-10-04. The independently frozen
[v31 pilot](nonlinear_load_v31_symbolic_results.md) passed numerical preflight;
paired learner tasks t132398/t132399 are running. v30 qualification and all
historical decisions remain unchanged.

## Limitations

Qualification is finite-horizon feasibility evidence, not an algorithm win or a stability certificate. The sustained cost is largely energy expenditure, not sustained state error or a bus-like latent regime. Learner robustness and superiority are unmeasured. Keep this new benchmark separate from all historical cohorts.

Evidence: `experiments/nonlinear_load_v30_environment_20261003/dispatch_manifest.json`; `results/nonlinear_load_v30_environment_qualification/{summary,block_analysis,task_observations}.json`.
