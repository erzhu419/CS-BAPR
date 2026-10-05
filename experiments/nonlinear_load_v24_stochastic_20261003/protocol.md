# Nonlinear load v24: stochastic continuation alignment

Reuse v22 last seed18 floor/no-floor exports t128679/t128680 and completed v23 initial-state/noise records. No training, fitting, new seeds, algorithm/environment changes or manuscript edits; historical HOLD remains unchanged.

## Frozen Measurement

One automatically selected Linux GPU runs both frozen models. Keep radii0.2/1, loads1/50, paths60000--60099, initial latent directions toward `-gain*observation`, first-step-only perturbations0/+/-0.01/+/-0.005, gamma0.99 and200-step native termination. Hold initial log std fixed under perturbation; later steps use each branch's own frozen stochastic actor, including its state-dependent std. Eight private CPU Torch generators92000--92007 supply common Gaussian draws to both models/all branches. Their first draws match v23 exactly; each subsequent draw has shape100x2. This is32,000 controller episodes, zero optimizer/fitting updates.

For each branch, return normalized physical reward `-discounted_cost/terminal_reward_std`; future entropy is `sum(t=1..199, gamma^t * -terminal_alpha * log_pi_t)`. Their sum is the finite-horizon soft Q return. Add initial entropy only for the unpenalized actor return. Native tanh density includes epsilon1e-6 and action range; no reward mean shift, terminal bootstrap or entropy after step199. Compare paired finite differences of actual returns against first-action critic mean, critic mean plus initial entropy, and risk-adjusted soft score. Keep risk separate; its ensemble std is not a measured return. Also record v23 analytic output-mean derivatives, finite-difference agreement, two-step sign reversals, per-path noise averages and eight per-draw block means. No significance/admission threshold is introduced.

## Execution And Next Decision

Four focused tests cover native sampling/density, analytic return/entropy timing and terminal handling, common noise/first-step shifts, compact statistics and unchanged parameters. One CUDA preflight uses radius1/load50, two paths and two draws (40 episodes). Require finite values, exact common initial states/external paths, return/cost decompositions and unchanged parameters before one formal task. Keep full per-draw/per-path evidence in separate remote raw directories; sync only compact result JSONs. Reuse existing small inputs; do not pull checkpoints/replay. No node/GPU preference, pin or allowlist; no Windows.

Study: `experiments/nonlinear_load_v24_stochastic_20261003`; results: `results/nonlinear_load_v24_stochastic_diagnostic`. If stochastic soft returns agree with critic ordering, interpret v23 as continuation/objective mismatch. If disagreement persists, localize physical versus future/initial entropy contributions before choosing one critic/target intervention. No new training is pre-admitted.

Execution: four frozen-runtime tests passed. Preflight t128950 exited0 and all40 episodes passed CUDA validation; only31,166 bytes returned. Formal t128953 exited0 in341.2 seconds on automatically selected node007/GPU0; all32,000 episodes passed the frozen validator, and1,781,921 bytes of compact results returned. Full raw matrices remain remote. The initial preflight t128884 failed before execution after three staging Broken pipe errors; its record is retained. Thin source/input transfer and reuse of the worker's existing wheelhouse restored dispatch without a placement preference or scheduler/runtime change. See `docs/nonlinear_load_v24_stochastic_results.md`: stochastic alignment does not remove three of four radius1 block-mean discrepancies. No follow-on training was submitted; historical HOLD is unchanged.

## Limitations

Eight action-noise draws approximate expectations. This finite-horizon snapshot freezes alpha/reward scale; it does not reconstruct nonstationary training, the pessimistic min-ensemble fixed point or OOD calibration. Local output-mean perturbations do not measure full actor/Jacobian/Adam updates. One reused seed/two radii is diagnostic evidence, not general RL superiority.
