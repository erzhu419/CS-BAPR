# v69: matched-state counterfactual reference screen

Frozen before dispatch on 2026-10-08. Keep v66 and the held v68 results.
Reuse v68's quadratic ridge models, full Gaussian densities, mean-log-density
window, phase-age calibration settings and original replay gates unchanged.

Use source v40 fixed-minus2 Hopper seed1 (99,745 updates) and v52 shared
context/replay. Independently start a frozen and an updating normal branch
for each role: fitting seed135040, density calibration seed135050, normal
null seed135060. Each branch lasts 6,000 steps at native TimeLimit 1,000.
Updating branches use only their actual normal transitions, fixed beta -2.

For every fitting/density-calibration transition, snapshot full MuJoCo
integration state, reset RNG state and TimeLimit age before the actual step.
Restore it in five fixed-profile shadow environments, then execute the same
action. All inputs are identical across modes. The normal shadow must exactly
reproduce observation, reward and termination/truncation. The shadows never
update policy/context/replay. Use actual normal trajectories for null
calibration, keeping frozen/updating stream histories separate.

Restore integration state directly before mj_step, without an extra
mj_forward that changes the solver warm start. The corrected restoration
passed 1,000 source-policy normal replay checks on native server MuJoCo 3.2.7.

Budget: 36,000 actual normal interactions plus 120,000 shadow simulator steps
= 156,000 total step calls. This includes 24,000 normal-shadow validation
steps. Fit and density calibration each have 12,000 labels per mode;
normal-null calibration has 12,000 observations. Reference-only normal policy
updates total 18,000; the source checkpoint and main controller stay unchanged.

Replay the same 12 v62 and 5 v61 cells. Normal first800/full/last1000/last64
mean risk must remain <=0.10; every changed first_post50/last64 paired contrast
must be positive. True-mode versus best-other mean log density must also be
positive in every changed pair/window before considering sequential belief.
No model selection or threshold tuning uses replay traces.

GPU chosen by scheduler without preference; server-only inputs currently
limit eligibility to node007. No Windows. Resume uses existing full-state
agent/RNG/context/environment checkpoints. Pull only scalar JSON/logs.

## Limitations

Known modes and full simulator state are privileged reference information.
This changes both counterfactual sampling and updating-policy coverage, so
it is not a single-factor ablation. Normal states need not cover later shifted
trajectories, and online observations still clip velocities. Reused traces
and one source training seed do not establish independent RL-return gains.
