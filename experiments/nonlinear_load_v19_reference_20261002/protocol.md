# Nonlinear load v19: matched actor reference replacement

v18 passed the frozen identification/direct-feedback gates. Train fresh seeds13/14 with the original linear reference versus cubic-fit local reference, four formal runs, without historical pooling.

## Single Intervention

Both arms collect the same per-seed severity1/radius0.2 random-action identification data (10,000 transitions), followed by1,000 validation transitions. Both fit the v18 linear and cubic models to that data at threshold0.001. Select only the frozen feedback gain used by the Jacobian penalty: original linear fit or the cubic fit's local action derivative at zero. Retain the unchanged auxiliary linear world model in both arms; the candidate does not consume its predictions during updates. Store both fit reports and gains in full checkpoints and result JSONs.

All other v17 reset-balanced settings remain fixed: centered tanh MLP, no NAU, no target shift or critic-symbolic penalty, Jacobian weight0.01, half ordinary replay/half stored-reset Jacobian states, independent sampler seed100000+training_seed, and balanced absolute-episode reset cycle0.2/0.5/1. Both train severity1 for500 episodes/100,000 interactions/99,001 updates. Identification does not consume learner replay; reset training RNGs after identification. Same initialization and update rules, with trajectories allowed to diverge after the reference changes actor updates. Save every20 episodes and evaluate last checkpoint only.

## Evaluation And Gates

Evaluate all three radii at severities1/50 on100 paired paths60000--60099, with actor, own fitted reference, known local and known inverse controls. Save early50/late150 state/command costs and initial/trajectory alignment. Per run2,400 controller episodes; no evaluation fitting or optimizer updates.

Reference repair requires strictly lower radius1 total and early state cost than the same-seed linear-reference actor at both loads. Small-radius retention requires radius0.2 total cost no greater than that actor at both loads, with zero margin. Own-reference added value requires cubic-reference actor cost strictly below its own fitted feedback at each radius, both seeds and loads. Proceed to learner-comparator replication only if all gates pass. Otherwise retain HOLD without retuning the reference, threshold, sampling ratio, weight, seeds or checkpoints. Historical v13/v15/v17 HOLD and manuscript tables remain unchanged.

## Verification And Dispatch

Focused tests must reproduce the original identification path, verify matched fit/gain metadata and unchanged initial critic/alpha updates, and prove resumed training preserves reference, metadata, sampler, RNGs and learner updates. Failures stop dispatch. Two six-episode CUDA preflights use seed13, one per reference; save/reload after episode5 before the final200 steps. Require201 updates,25,728 reset plus25,728 ordinary Jacobian samples, matched first-five exploration histories and both fit reports/gains, and equal known-control returns. Preflights are not efficacy evidence.

Scheduler selects idle compatible Linux GPUs without preference, pin or allowlist; no Windows. Full checkpoints stay on servers outside output directories. Pull only small logs/JSONs. No runtime optimization or paper edits.

## Limitations

This is reference replacement in one disclosed nonlinear environment, not general RL superiority. Cubic basis motivation uses the disclosed actuator but coefficients are data-fitted. v18 retained a pruned positional load coefficient; its threshold is frozen. Improved direct feedback does not guarantee actor added value, and direct controllers are not certified optima. Reused disturbance paths and two fresh seeds bound inference.
