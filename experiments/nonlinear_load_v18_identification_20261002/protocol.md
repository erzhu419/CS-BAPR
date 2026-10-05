# Nonlinear load v18: matched-data actuator identification

v17 improved initial reference alignment but retained HOLD. Check whether omitted cubic action terms explain the seed-dependent load gains before changing actor training.

## Frozen Diagnostic

Use the v17 numerical runtime and seeds11/12. Recreate the original severity1/radius0.2 collection (50 episodes, 10,000 transitions), followed by its five-episode validation. Require original fitted feedback gains to reproduce within1e-6 absolute error and the original full-state R-squared within1e-10. A reproduction failure stops interpretation.

Fit both models to the same transitions, with unchanged STLSQ threshold0.001 and a degree-one library over state/control inputs. The control has inputs u_x/u_y. The intervention adds only u_x^3/u_y^3 as additional control features, without interactions or higher-order state terms. No true dynamics coefficients are used in fitting. Construct both references by the same discrete Riccati/lstsq rule; for the cubic fit, use its fitted linear action coefficient as the local derivative at zero. References remain tanh linear feedback, not inverse-actuator policies.

For each seed, independently collect five episodes at each severity1/50 with radius0.2 and random actions, using seeds71000+training_seed and72000+training_seed. Score physical state increments only; stochastic next-load innovations are not a physical-dynamics error. Report matrices and errors against known environment coefficients for diagnosis only.

Evaluate original fitted, cubic-local fitted, known local and known inverse controllers at radii0.2/0.5/1 and severities1/50 on100 paired paths60000--60099. Require the three existing controller return arrays to reproduce. Save total/early/late state and command costs. Per seed: 10,000 recreated identification, 1,000 original validation, 2,000 independent validation interactions, 2,400 controller episodes. No actor/critic optimization or checkpoint input/output.

## Decision

Proceed to a fresh matched actor experiment only if both seeds have lower physical prediction RMSE at both independent validation loads, recovered local load gain with maximum absolute error<=0.001 relative to diag(0.4,0.4), and strictly lower direct-feedback total cost in every radius/load block. Keep each seed separate. Failure retains HOLD; no post-hoc library/threshold/data/controller retuning. Historical v13/v15/v17 HOLD and paper tables remain unchanged.

## Operations And Limitations

Focused tests check coefficient/local-gain recovery, unchanged fit data, physical-only validation and retention of individual seed/block failures. Scheduler uses idle compatible Linux GPUs without preference, pin or allowlist; no Windows. Reuse frozen dependencies and stage only compact source-controller JSONs. Return small diagnostic JSONs; no CSV/replay/checkpoint pulls.

This is a known-environment model diagnostic, not general RL superiority or an optimal-controller certificate. Cubic features are motivated by the disclosed actuator; gains are identified from data. Direct-reference improvement cannot establish actor added value. Reused disturbance paths and two identification seeds limit inference.
