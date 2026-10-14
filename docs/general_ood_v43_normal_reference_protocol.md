# v43 Independent Normal Reference Probe

Frozen 2026-10-06 after v42 HOLD. Keep v40 seed1 fixed-minus2/relative-adaptive policies frozen. Reuse v42's three predictor capacities, ridge mean-loss coefficient 1e-3, polynomial features and scale floors. No RL updates, replay writes, NAU/SINDy or changed physics profiles.

## Reference Split And Coverage

Per policy, collect normal-only fitting seeds 98000-98007 and calibration seeds 98010-98013, each 3072 transitions, TimeLimit 1536. Restore the same training context before every rollout; physical termination/reset and online context remain active. This covers full episodes and resets instead of just a single 512-step fitting prefix. Fit the predictor once using all fitting trajectories; fit the residual calibration on separate trajectories. Never use test transitions for either stage.

Three fixed calibrations share each predictor: global-rms (v42 formula), global-centered (subtract normal residual coordinate means, scale by centered RMS), episode-centered (the same centering/scaling separately by current episode step 0-255, 256-511, 512-767, 768+). Use each calibration group's empirical score95 threshold. Episode step is known before the current transition; neither current/future termination nor mode labels select its group. Include real terminal targets, never difference across resets. Compute episode ages separately before concatenating independent trajectories.

## Paired Tests

Use the identical v42 development seeds 90000-90019, five modes, 1536 interactions, intervention before 1024 and the unchanged 1536 TimeLimit. Verify the entire pre-switch prefix and state/executed-action/context at the first changed transition. Restore context per rollout. Report validation256, first_post16, first_post256 and second_post256 with calibrated mean/first-transition scores, exceedance fractions, nonterminal means and explicit termination/truncation counts. Retain all predictors/calibrations, not the best OOD arm. Record fitting/calibration error and test forecast error separately from calibration scores. Compare no-change alarm fractions with v42 only after exact reward/ending-trace reproduction.

Formal totals: 24 reference rollouts, 73728 reference interactions; 200 paired tests, 307200 interactions; 380928 total. Independent reference trajectories are new, but the test cohort remains development data, not independent confirmation.

## Execution And Judgment

CUDA preflight uses separate normal fitting seeds 97900-97901, calibration 97910-97911; tests 99000-99001 in normal/high_grav. Eight reference rollouts plus eight tests, 36864 total interactions. Require disjoint splits, all four observed calibration age groups, frozen source counters, finite statistics and v42 trajectory reproduction. Settings do not change based on preflight OOD scores.

Run on the data host node007 without preferred/required GPU flags; no Windows or checkpoint transfers. Pull only compact JSON/logs; full trajectories stay in worker memory. Inspect no-change forecasts/calibration first, then early change response before falls. Poor null forecasting or weak early response retains HOLD for RL feedback. Broader fitting and calibration coverage change together; this diagnoses their combined effect without isolating their separate contributions.

## Limitations

Two frozen policies from one training seed; deterministic policy/state coupling, polynomial capacity and clipped observations remain. Episode age is only a coarse observable-history proxy, not a general state-conditioned dynamics model. Empirical calibration95 need not produce a 5% test alarm rate. No extreme50x return evaluation or RL policy-improvement claim.
