# v39 surprise-scale diagnostic

Frozen 2026-10-06. v38 remains HOLD: Pendulum average ID/OOD gains did not pass its fixed-profile floor condition, and Hopper corrected seed1 diverged before any Markov OOD exposure.

## Intervention And Controls

Hopper-v4, training seed1 only, four paired arms on one CUDA runtime:

- `raw-adaptive`: corrected v38 belief with the original absolute critic L1-drift surprise.
- `relative-adaptive`: replace only that signal with `0.5 * mean(abs(L1_current-L1_target)/max(abs(L1_target),1e-6))`, computed per critic before averaging.
- `fixed-minus2`: relative surprise, penalty_scale=0, beta=-2.
- `fixed-minus3`: relative surprise with lambda fixed at1; beta=-3 after the same 100-update beta=-2 warmup as the adaptive arms.

Reward z-score, one-sided Q-std ratio, max aggregation, EMA, belief likelihood, actor/context, critic objective, reward scaling, entropy settings and learning rates remain matched. The normalization divides by each target critic's norm, not a Hopper-specific tuned coefficient. The default raw surprise algorithm is unchanged; its instrumentation is tested against frozen v38. NAU/SINDy/Jacobian losses remain absent.

## Budget And Measurements

Exactly 100000 training interactions per arm, mode switches every5000 interactions using the same independent seed1 clock. No early stopping or structured50x training exposure. This is a short mechanism screen, not a converged Hopper benchmark. One optimizer update per interaction after batch256 is available gives99745 updates. Budget cutoff does not create an MDP terminal.

Record 1000-step means of all surprise components, their dominant-source fractions, reward normalization std, policy entropy, critic L1 norms/Q values, alpha, losses, belief reset/window and effective beta. Diagnostic means are update-weighted when combined. Recent completed-episode returns track policy deterioration. Means reveal sustained saturation, not individual shock timing.

Select only the final checkpoint, restoring training context normalizers. Evaluation remains the v38 protocol:20 paired seeds starting80000; ID, structured Markov50x with period200/probability1, and five episodes per non-ID fixed profile. Each evaluation block starts from the same training context snapshot. No optimizer or belief updates during evaluation. Record actual non-ID transition counts, including zero exposure if episodes terminate before the first switch.

Run a four-arm 512-interaction CUDA preflight first, with two primary/one fixed-profile episode. Submit the formal bundle only after all four exits, metrics, budgets, checkpoint restoration and paired configurations/runtime pass. Scheduler chooses among compatible Linux nodes, with no preferred/required node; checkpoints stay remote, only compact JSON/logs return locally. Interrupted jobs restart from scratch.

## Decision And Limits

Diagnose which component dominates raw surprise and whether relative calibration restores a non-saturated belief trajectory. Compare relative-adaptive with raw-adaptive and both constant coefficients at the same100k budget. Advance to a two-seed, original-budget screen only if relative-adaptive improves ID and exposed Markov OOD over raw-adaptive, matches or improves the best constant control on both returns and the worst fixed-profile floor, and avoids sustained alpha/Q/loss escalation. If raw failure is not reproduced at100k, do not attribute its300k collapse to the normalized component.

Any favourable result is seed1 development evidence. Fixed-minus3 tests penalty strength, not a new RE-SAC implementation. The environment, library runtime and evaluation protocol are unchanged; v38's300k endpoint must not be compared as an equal-budget control. No manuscript claim changes before broader support.
