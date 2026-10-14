# v52: matched warm continuation

Four variants share the v40 fixed-minus2 seed1 final checkpoint, including its
actor/critic/target and optimizer states. Network architecture, context, replay
sampling and all losses remain identical; only the actor beta coefficient changes:
fixed-minus2=0, fixed-minus2p3=.3, noise-age=v50 noise-likelihood age coefficient,
age-departure=v51 normal-age calibrated departure. Beta=-2-coefficient, scale=1.
The checkpoint's 99745 updates already exceed the inherited 100-update warmup.

Prepare one shared normal-only dataset per stage using stochastic actions and
1000-step TimeLimit, matching training. Fit quadratic state/executed-action
delta-state ridge on 3072 steps per fit seed; calibrate global-centered residual
score, startup noise and unchanged v51 age thresholds on separate calibration
seeds. Only fit transitions refill replay. Restore checkpoint context for each
reference rollout and at each arm's start; reward EMA starts at 1 for every arm.
Normal null trajectories are excluded from replay, fitting and calibration.

Preflight fit97920/1, cal97930/1, null97940/1; formal fit98100--107,
cal98110--113, null98120--123. Preflight adds6000 interactions/updates per arm;
formal adds30000. Shared normal costs18432/49152 respectively, counted once.
Use identical RNG seed1 and environment reset seeds, and v38 mode redraws every
5000 steps. Update readouts once per real transition, before its optimizer update;
never reset their history or refit predictors during continuation.

Evaluate restored final checkpoints only: ID, structured Markov50x switching every
200 steps, and all four non-ID static profiles. Preflight uses2/1 episodes from
99520; formal20/5 from91500. Restore training context before each evaluation
episode. Paired bootstrap intervals condition on these trained policies only.

Four arms run concurrently in one scheduler bundle and share the same server-side
replay/reference file. Existing source data are on node007; scheduler selects its
idle GPU without preference, no Windows nodes. Return scalar JSON/logs only.
Pass scheduler `--stage-exclude run_shared` so source staging preserves this
server-generated shared package when later diagnostics reuse the same cwd.
Full continuation checkpoints save replay, RNG, context/readouts, counters and
native MuJoCo integration state every5000 steps and at completion. They remain
server-side. The v40 source lacks replay: this is warm continuation, not its exact
resume. Keep preflight returns out of parameter selection; retain negative results.
