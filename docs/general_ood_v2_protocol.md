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

## Training

Train under the generic stressor:

```bash
python scripts/train_csbapr.py \
  --env Pendulum-v1 \
  --method csbapr \
  --seed 0 \
  --max-episodes 50 \
  --extreme-ood-train \
  --ood-schedule episode_random \
  --ood-intensity 50
```

For MuJoCo:

```bash
python scripts/train_csbapr.py \
  --env Hopper-v4 \
  --method csbapr-no-sindy \
  --seed 0 \
  --extreme-ood-train \
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

## Evaluation

Evaluate random OOD intensity sweeps:

```bash
python scripts/ood_eval.py \
  --env Pendulum-v1 \
  --extreme-random \
  --range 1,5,10,20,50,100 \
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
