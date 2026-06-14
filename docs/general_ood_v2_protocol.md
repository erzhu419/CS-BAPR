# CS-BAPR v2: General Extreme-OOD RL Protocol

This branch starts the v2 direction: move beyond the bus-only result by
testing the same algorithmic idea under bus-like extreme OOD randomness in
standard RL environments.

## Goal

The v1 paper shows the strongest result on the bus environment, where OOD is
not a small perturbation but a very large demand multiplier plus abrupt and
oscillating shifts. For v2, the benchmark should expose the same stress pattern
to generic Gymnasium environments:

- static magnitude OOD,
- mid-episode abrupt shifts,
- oscillating non-stationary shifts,
- per-episode random OOD modes,
- within-episode random OOD mode changes.

## Implementation

The generic wrapper is `csbapr.envs.extreme_ood.ExtremeOODWrapper`.

It applies perturbations in two layers:

- Physics layer, when available:
  - Pendulum: `mass`, `gravity`, `length`.
  - MuJoCo: `body_mass`, `friction`, `gravity`.
- Fallback layer, available for any Box-observation/Box-action Gym env:
  - observation scaling and noise,
  - action gain and noise,
  - optional reward scaling and noise.

This means the same protocol can run on Pendulum, Hopper, Walker2d,
HalfCheetah, Ant, or a non-physics custom Gym environment.

## Profile Families

`--ood-profile-family extreme`
: The original generic stress test. It deliberately mixes severe physics,
sensor, and actuator perturbations. This is useful for finding failure modes,
but it is less analogous to the bus benchmark because the OOD source is not a
single coherent latent regime.

`--ood-profile-family structured`
: A bus-like latent-load family. Each profile is a coherent regime such as
`low_load`, `rush_load`, `mega_event`, or `slippery_rush`; the profile changes
mass/gravity/friction together instead of injecting unrelated sensor/action
chaos. The agent does not observe the profile label. The label and
`latent_load` are only emitted in `info` for diagnostics.

`--ood-profile-family load`
: The current recommended v2.2 family. It keeps the coherent latent regime
idea, but makes the OOD more bus-like: a hidden load variable jointly changes
physics, actuation, terrain/contact, runtime external force, and locomotion
velocity preference. The intensity is still allowed to be large, but is mapped
to severe-yet-solvable profiles instead of blindly making every parameter 50x.

`--ood-profile-family controllable_load`
: The v2.3 Pendulum diagnostic family. It keeps the hidden latent-load
interface but makes the dominant disturbance an additive actuator/load torque,
with only mild mass/gravity/length changes. This is closer to the bus
benchmark's structure: the OOD regime is severe and hidden, but an
approximately linear compensating action remains feasible. Use this instead of
plain `load` on Pendulum, where the previous high-load profile mostly measured
torque saturation.

## Schedules

`--ood-schedule static`
: One OOD profile is held for the full episode.

`--ood-schedule episode_random`
: One severe non-ID profile is sampled at each reset.

`--ood-schedule abrupt`
: The episode starts ID and switches once at `--ood-abrupt-step`.

`--ood-schedule oscillating`
: Profiles cycle every `--ood-switch-steps`.

`--ood-schedule step_random`
: A new severe profile is sampled every `--ood-switch-steps`.

`--ood-schedule markov`
: Start ID, then at each `--ood-switch-steps` tick switch profiles with
`--ood-switch-probability`. This creates longer coherent dwell times, closer to
bus traffic/load regimes than fully random step switching.

## Training

Train under the generic stressor:

```bash
python scripts/train_csbapr.py \
  --env Pendulum-v1 \
  --method csbapr \
  --seed 0 \
  --max-episodes 50 \
  --extreme-ood-train \
  --ood-profile-family structured \
  --ood-schedule episode_random \
  --ood-intensity 50
```

Recommended v2.2 latent-load run:

```bash
python scripts/train_csbapr.py \
  --env Hopper-v4 \
  --method csbapr-stable-context \
  --seed 0 \
  --extreme-ood-train \
  --ood-profile-family load \
  --ood-schedule markov \
  --ood-intensity 50 \
  --ood-switch-steps 200 \
  --ood-switch-probability 0.35
```

For MuJoCo:

```bash
python scripts/train_csbapr.py \
  --env Hopper-v4 \
  --method csbapr-no-sindy \
  --seed 0 \
  --extreme-ood-train \
  --ood-profile-family structured \
  --ood-schedule abrupt \
  --ood-intensity 50 \
  --ood-abrupt-step 250
```

For contact-rich MuJoCo tasks, start with `csbapr-no-sindy`, `csbapr-no-jac`,
or `csbapr-no-sym`; SINDy assumptions are usually poor under contacts.

## v2.1: Online Context Adaptation

The first extreme-OOD MuJoCo/Pendulum matrix showed that direct
`csbapr-no-sindy` transfer does not reproduce the bus result. This is an
important negative result: generic continuous-control tasks do not expose the
same route/demand structure that the bus environment gives CS-BAPR.

The first algorithmic adaptation in this branch is therefore not to weaken the
OOD wrapper, but to give the agent a non-oracle latent regime descriptor:

- `csbapr-context`: NAU actor + BA-PR losses + online transition context.
- `bapr-context`: MLP actor + BA-PR losses + the same online transition context.

The context is inferred from recent `(s, a, r, s')` statistics:
`delta_s`, `|delta_s|`, `action`, `reward`, `||delta_s||`, and `||action||`.
It is appended to actor/critic inputs and stored with each replay transition.
This makes off-policy training regime-aware without using the wrapper's
`ood_profile` label.

Use `bapr-context` as the key ablation: if it improves as much as
`csbapr-context`, the gain is from online regime inference rather than NAU.

The initial context-adaptation runs did not recover the bus-level CS-BAPR
advantage under the unstructured `extreme` family. The next protocol therefore
separates two questions:

- `extreme`: robustness to arbitrary compound chaos.
- `structured`: robustness to bus-like latent regime shifts with very large OOD
  magnitude.

## v2.2: Latent-Load Environment and Diagnostics

The structured matrix showed an additional failure mode: raw context variants
could learn early on Hopper but collapsed late, while no-context baselines
remained stable. The immediate fix is not more random search. The training
script now:

- passes `recent_rollout` into `agent.update(...)`, enabling rollout-based
  BAPR surprise instead of replay-batch surprise;
- writes recent training diagnostics (`surprise`, `belief_entropy`,
  `weighted_lambda`, `q_std`, `alpha`, `L_eff`) into the history JSON;
- records OOD profile switch events;
- runs fixed-profile evals for each wrapper profile, so worst-regime return can
  be inspected separately from random-schedule average return.

New conservative context baselines are available:

- `csbapr-stable-context`
- `bapr-stable-context`
- `csbapr-no-sindy-stable`

They slow the transition-context EMA and strengthen behavior cloning regularity
to reduce context-triggered actor drift. The CS-BAPR stable variants also cap
entropy alpha, clamp the Gaussian mean before tanh, and penalize a
differentiable NAU/NMU L_eff surrogate.

## Evaluation

Evaluate random OOD intensity sweeps:

```bash
python scripts/ood_eval.py \
  --env Pendulum-v1 \
  --extreme-random \
  --range 1,5,10,20,50,100 \
  --ood-profile-family structured \
  --ood-schedule abrupt \
  --ood-abrupt-step 100 \
  --seeds 10 \
  --output results/pendulum_extreme_random.json
```

With trained checkpoints:

```bash
python scripts/ood_eval.py \
  --env Hopper-v4 \
  --extreme-random \
  --range 1,5,10,20,50 \
  --ood-profile-family structured \
  --ood-schedule oscillating \
  --ood-switch-steps 100 \
  --checkpoint-nau results/csbapr_Hopper-v4_0_best.pt \
  --checkpoint-relu results/bapr_Hopper-v4_0_best.pt
```

## First v2 Matrix

Start small and reproducible:

| Env | Primary methods | Schedules | Intensity |
| --- | --- | --- | --- |
| `Pendulum-v1` | `csbapr-context`, `bapr-context`, `csbapr-no-sindy`, `bapr`, `resac` | step_random, abrupt | 10, 50 |
| `Hopper-v4` | `csbapr-context`, `bapr-context`, `csbapr-no-sindy`, `bapr`, `resac` | step_random, abrupt | 10, 50 |
| `Walker2d-v4` | `csbapr-context`, `bapr-context`, `bapr` | step_random | 10, 50 |

The key v2 question is not whether NAU always wins. It is whether the six
training-stabilization fixes plus architecture choices produce a robust family
under severe non-stationarity, and which environments actually match the
constant-derivative inductive bias.

## Structured-OOD Follow-Up Matrix

After the negative `extreme` and context-adaptation results, the most relevant
next matrix is:

| Env | Methods | Schedule | Intensity |
| --- | --- | --- | --- |
| `Pendulum-v1` | `csbapr-no-sindy`, `bapr`, `resac`, `csbapr-context`, `bapr-context` | step_random | 50 |
| `Hopper-v4` | `csbapr-no-sindy`, `bapr`, `resac`, `csbapr-context`, `bapr-context` | step_random | 50 |

Interpretation:

- If CS-BAPR improves only on `structured`, the contribution is not generic
  chaos robustness; it is compatibility between the architecture and
  low-dimensional coherent OOD regimes.
- If BAPR/context variants close the gap, the paper should emphasize online
  regime inference rather than NAU.
- If all methods remain weak, the result is a clean negative result: bus has
  environment-specific structure that generic continuous-control tasks do not
  reproduce under current wrappers.

## Latent-Load Follow-Up Matrix

Use this matrix before launching larger sweeps:

| Env | Methods | Schedule | Intensity |
| --- | --- | --- | --- |
| `Pendulum-v1` | `resac`, `bapr`, `csbapr-no-sindy`, `bapr-stable-context`, `csbapr-stable-context` | markov | 50 |
| `Hopper-v4` | `resac`, `bapr`, `csbapr-no-sindy`, `bapr-stable-context`, `csbapr-stable-context` | markov | 50 |

Primary readouts:

- fixed-profile worst return,
- post-switch recovery under `profile_events`,
- whether `surprise` and `weighted_lambda` move after profile switches,
- seed variance of the stable-context variants.

## Seed-0 Latent-Load Readout

The first seed-0 latent-load run produced a useful split:

- Hopper: `bapr-stable-context` improved worst fixed-profile return over
  `bapr` and `resac`, so latent context is promising.
- Hopper: NAU variants (`csbapr-no-sindy`, `csbapr-stable-context`) suffered
  late collapse with exploding alpha/q-std/L_eff, so do not expand CS seeds
  until the NAU stability knobs are validated.
- Pendulum: `switch_every_steps=200` equals the episode horizon, so it did not
  test mid-episode adaptation; use `--ood-switch-steps 50`.
- Pendulum high-load profiles were too hard, clustering all methods near the
  minimum return; the load profile has been weakened for the next diagnostic.

Next diagnostic matrix:

| Env | Methods | Seeds | Schedule |
| --- | --- | --- | --- |
| `Hopper-v4` | `resac`, `bapr`, `bapr-stable-context` | 1, 2 | `load + markov + switch_steps=200` |
| `Hopper-v4` | `csbapr-no-sindy-stable`, `csbapr-stable-context` | 0 | `load + markov + switch_steps=200` |
| `Pendulum-v1` | `resac`, `bapr`, `csbapr-no-sindy-stable`, `bapr-stable-context`, `csbapr-stable-context` | 0 | `load + markov + switch_steps=50` |

## v2.3: Controllable Pendulum Load

The follow-up matrix above confirmed two separate issues:

- Hopper: the NAU stability knobs worked. `csbapr-stable-context` recovered
  from the legacy collapse and reached a seed-0 fixed-profile worst return of
  878.4 under `load + markov`, comparable to the best non-NAU runs.
- Pendulum: `load + markov + switch_steps=50` still did not reproduce the bus
  effect. The fixed worst profile stayed dominated by `mega_event`, and the
  NAU/context variants underperformed `resac`/`bapr`.

For Pendulum, treat this as an environment-design failure rather than a final
algorithmic result. The new `controllable_load` family keeps exaggerated hidden
regime switches but makes the dominant OOD effect a recoverable action-bias
disturbance. The next diagnostic matrix is:

| Env | Methods | Seeds | Schedule |
| --- | --- | --- | --- |
| `Pendulum-v1` | `resac`, `bapr`, `bapr-stable-context`, `csbapr-stable-context` | 0, 1, 2 | `controllable_load + markov + switch_steps=50` |
| `Hopper-v4` | `csbapr-stable-context`, `csbapr-no-sindy-stable` | 1, 2 | `load + markov + switch_steps=200` |
