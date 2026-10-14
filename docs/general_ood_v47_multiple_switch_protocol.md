# v47: paired second-switch frozen replay

## Question

Does v46 global-centered, moment-matched Student df4 belief respond to returning
to normal and to a different OOD mode, after already adapting to an OOD level?
Keep episode-centered original adaptive Student as the control. No actor update.

## Fixed design

- Same two v40 Hopper seed1 policies, same v43 normal-only reference fit/calibration.
- First switch before transition 1024: normal -> heavy or normal -> slippery.
- Second switch before transition 1280: stay, return normal, or swap heavy/slippery.
- Budget 1792 transitions; retain TimeLimit 1536, physical termination and deterministic
  episode reset seeds. The second switch does not coincide with the fixed TimeLimit.
- For each policy/seed/initial mode, compare changed branches to one shared stay
  branch. Verify all pre-1280 arrays, filter histories and current switch inputs
  match exactly. The auxiliary observer never sees mode labels or controls actions.
- Same signal transform, H20, hazard .02, moment parameters, df4, uniform initial
  belief and exponential age-to-penalty mapping. No reset at episode boundaries.
- Preflight: two evaluation seeds per policy; formal: existing 90000--90019 per policy.
  Four changed plus two stay rollouts per seed, not independent filter-arm rollouts.
- References and legacy normal prefixes must match v46. In formal runs, stay
  trajectory summaries, residual scores and overlapping belief windows must match.

## Readout

Signed changed-minus-stay penalty, reset probability and age in first16/64 and
successive 256-transition windows, plus last64 recovery, all majority-reset events,
and termination/truncation counts. A downward signal change is not assumed to
require a positive penalty response. No hard-reset threshold is a universal success gate.

Preflight 67584 interactions; formal 503808, including normal references. Only
compact summary/event JSON and logs return locally. Existing checkpoints stay on
node007; the scheduler chooses an idle GPU there. No Windows nodes, no new training
seeds, environment tuning, actor training or manuscript changes in this round.
