# v129 frozen heldout-noise robustness

Use every training seed 0/1/2 and three predeclared final models: v123 300k
reference, v128 350k unchanged control, v128 350k mean retention. No training,
optimizer updates, readout refit, best-checkpoint selection or reward/physics
change. Normal native Hopper-v4, CPU inference and the 1000-step time limit.

Five common resets 150000--150004 and twenty new noise seeds 13000--13019.
Use v126's independent CPU torch.Generator with one float32 (1,3) draw per step,
full noise scale one, identical schedules for every model. Each source starts
with its native context normalizer, resets EMA per episode, updates context on
actual transitions, then restores context/RNG. Resets are reused, not heldout.

Each of three CPU tasks runs 315 episodes: three models times five deterministic
and 100 sampled episodes. Formal total: nine models / 945 episodes / at most
945000 evaluation interactions. First reproduce all 15 deterministic baselines
per task from v126 (reference) and v128 (control/retention); fail on outcome or
material scalar mismatch. Seed2 native smoke uses one reset and one new noise
per model at the full time limit: six episodes, three baseline matches. Smoke
tests execution, not model selection; formal protocol stays fixed.

Report every training seed and noise-wise paired retention-minus-control and
retention-minus-reference differences. Noise/reset repeats are not additional
training seeds. Examine whether the benefit persists beyond v128's small noise
groups, including regressions; do not pick a model or retune from this heldout
set. This round supplies no OOD or submission-readiness claim.

Freeze the v128 runtime plus the v129 runner; no core/training changes. Source
models, baseline JSON and simulator/runtime are native on Linux node007. No
machine/GPU preference or Windows nodes; tasks request CPU resources only.
Sync compact heldout.json (episode scalars and final health state), no traces,
CSV or checkpoints. Native progress.json supports episode-boundary resume.
