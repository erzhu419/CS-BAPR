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
| `Pendulum-v1` | `csbapr`, `bapr`, `sac`, `dr` | static, abrupt, oscillating | 1, 5, 10, 20, 50, 100 |
| `InvertedPendulum-v4` | `csbapr-no-sindy`, `bapr`, `dr` | abrupt, step_random | 1, 5, 10, 20, 50 |
| `Hopper-v4` | `csbapr-no-sindy`, `csbapr-no-nau`, `bapr` | abrupt, oscillating | 1, 2, 5, 10, 20, 50 |

The key v2 question is not whether NAU always wins. It is whether the six
training-stabilization fixes plus architecture choices produce a robust family
under severe non-stationarity, and which environments actually match the
constant-derivative inductive bias.
