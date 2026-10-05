# Nonlinear load v12: frozen-policy new-path confirmation

Evaluate the eight completed v11 policies: Jacobian-only, no-symbolic, BAPR and RE-SAC, each with training seeds 7/8. Keep the exact validated v11 runtime, actor parameters, fitted gains, environment and six severities 1/2/5/10/20/50. Change evaluation paths only, to 60000--60099; no existing local evaluation artifact uses these paths. No training, fitting, loss adjustment or checkpoint selection is allowed.

Export only float32 actor parameters, saved reference gains and source metadata into small JSON inference inputs on the original workers. Full checkpoints, replay buffers and optimizer states remain there. Verify tensor round-trip equality and source completion/configuration before export. Portable inputs let the scheduler select idle Linux/WSL resources without node preferences, allowlists or GPU pins; Windows is prohibited.

Each task first reproduces old path 50000 at severities 1 and 50 for its actor and all controls. Reuse the prior diagnostic tolerances: rtol=1e-4, atol=1e-3. Failure stops evaluation before new paths. Then evaluate the actor, known local-linear and inverse-actuator controls on all 100 new paths at all six severities. Candidates additionally evaluate their own saved fitted feedback, without refitting. Reuse the original serial deterministic CUDA evaluation routines and pinned packages. Save paired raw returns, full source configuration, saved gains, runtime versions and reproduction checks; return only small result JSONs.

Apply the same gates to the new paths alone: lower mean cost and 2/2 matched-training-seed wins at ID and 50x against each of no-symbolic/BAPR/RE-SAC, plus positive learner-minus-own-fitted mean returns for both seeds at ID and 50x. Old paths and development training seeds 5/6 cannot offset a failure. Retain worst paired gaps. This is new disturbance-path confirmation in the same environment, not a new-environment or general-RL claim.

Formal evaluation contains 15,600 episodes including controls; reproduction checks add 52 episodes on an already-used path. All new learner and identification interactions and optimizer updates are zero. Preserve all historical gates and manuscript tables. Interrupted evaluation may restart from its immutable input; no training checkpoint is created or resumed.

## Dispatch

Two CUDA preflights (t121296/t121297) and two focused unit tests passed. The largest preflight return deviation was 9.14e-6. Inference inputs pulled from remote workers total 729,795 bytes; no full checkpoint was downloaded. Formal tasks t121303--t121310 have all started and passed their own reproduction checks. Initial placement: six tasks on node007 GPU1/2/3, one on jtl110gpu2 GPU0 and one on local GPU0, chosen without preferences. Detailed task/source mappings and startup observations are saved in the v12 manifest and `results/nonlinear_load_v12_paths/startup_observations.json`.

## Completion

All eight tasks finished with exit code 0. The new-path cohort independently passes all three learner-comparison gates and the own-fitted-feedback gate. Source configurations, saved gains, zero updates/fitting, reproduction values and all paired raw returns were verified. Results and limitations are in `docs/nonlinear_load_v12_paths_results.md` and `results/nonlinear_load_v12_paths/summary.json`. Three summary tests passed. No further tasks were submitted in this completion turn.
