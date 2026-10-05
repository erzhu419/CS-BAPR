# v37 route-B cross-environment protocol

Frozen 2026-10-04 (Asia/Shanghai). This is the first route-B protocol intended
to support a generic-OOD claim. Earlier v2 results remain historical because
they mix environment versions, OOD families, checkpoint-selection rules and
budgets.

## Matrix

Environments are `Pendulum-v1` and `Hopper-v4`, representing low-dimensional
single-body control and contact-rich locomotion. Each environment uses the same
four methods:

- `csbapr-stable-context`: candidate method with online transition context;
- `csbapr-no-sindy-stable`: matched no-context NAU ablation;
- `bapr-stable-context`: context-only predecessor control;
- `resac`: risk-sensitive SAC baseline without BAPR adaptation.

Each method/environment pair uses training seeds 0, 1 and 2. Training uses the
existing ID-mode training entry point and the environment preset's fixed budget;
no OOD profile label is exposed during training or evaluation.

## Evaluation

The final checkpoint is evaluated on 20 paired evaluation seeds starting at
70000. OOD uses `ExtremeOODWrapper` with `structured` profile family,
intensity 50, and a Markov schedule that switches with probability 1.0 every
50 Pendulum steps or 200 Hopper steps. The profile label is retained only in
the output diagnostics. Every method also receives five fixed-profile episodes
for each non-ID profile, so the worst regime cannot be hidden by the Markov
average.

The evaluator performs zero optimizer updates. Only compact JSON and logs are
returned; checkpoints stay on the worker. The primary comparisons are paired
ID and 50x Markov return/cost against BAPR and RE-SAC. The mechanism readout is
the matched `csbapr-stable-context` versus `csbapr-no-sindy-stable` comparison.

## Admission rule

Route B is not admitted from one favorable environment. The candidate must
beat both BAPR and RE-SAC on the three matched training seeds in both ID and
50x OOD for at least one environment, while the second environment must not
show a catastrophic failure or an unreported seed collapse. The ablation is
reported even if it fails; it is not tuned after seeing the result.

This protocol does not claim universal RL transfer. If the gate fails, the
correct result is a cross-environment negative or boundary result, and the
paper claim remains environment-conditional.

## Interpretation correction (2026-10-05)

The original protocol above is preserved; runs and presets are unchanged.
All four presets disable SINDy and Jacobian losses. The no-context preset
also changes `beta_bc` from 0.005 to 0.001 and `penalty_scale` from 1 to 2,
so it is a multi-setting control, not a single-factor context ablation.
Hopper uses an equal episode limit, not equal interactions, because early
termination is policy-dependent. The evaluator records returns and lengths,
not a separate Hopper cost. v37 can diagnose transfer but cannot establish
the full symbolic mechanism or an interaction-budget-matched generic claim.
