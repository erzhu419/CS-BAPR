# OOD Extreme Scenarios — Archive and Migration Notes

## Why this document exists

While planning the "震荡 non-stationary OOD" experiment for CS-BAPR on
MultiLineEnv, a search through the repo surfaced a richer OOD-mode
specification in `bapr_reference/mode_profiles.py` that we used earlier
in the BAPR work. The current `offline-sumo/env/sim_core/sim.py` path
(MultiLineEnv) only supports two OOD levers:

1. **`od_mult`** — a single global multiplier on passenger arrival rates,
   fixed for the entire episode.
2. **`set_ood_burst(inject_time, burst_mult)`** — a single step at
   `inject_time` that switches the global multiplier for the remainder
   of the episode.

The legacy BAPR OOD modes below are strictly richer. This file documents
them so the information isn't lost, even though the current experiment
batch on JTL110GPU2 uses only the simpler `od_mult` + single-burst
levers. See the "Migration plan" section for what would need to move
into MultiLineEnv.

## Current status (2026-04-23)

* Current CS-BAPR experiments running on JTL110GPU2 use only
  `od_mult ∈ {1, 2, 5, 10, 20, 50}` (static sweep) and single-step burst
  at `t = 3600s` with `burst_mult ∈ {5, 10, 20, 50}`.
* The **multi-burst within-episode schedule** (plan option B1) is being
  added next: a new `set_ood_schedule([(t, mult), ...])` method on
  env_bus and MultiLineEnv, implementing a piecewise-constant demand
  profile over time. This is the most minimal extension toward a truly
  non-stationary OOD scenario.
* The **full legacy OOD modes** (plan option B2) — station-level OD
  overrides, speed scaling, affected-route subsets, qualitatively named
  scenarios — are **not** being ported in the current batch. They
  remain as a potential future experiment if the paper wants the "real
  operational extremes" figure rather than the "parametric sweep" figure.

## Legacy OOD_MODES (from `bapr_reference/mode_profiles.py`)

These are designed for the old `bapr_reference` env_bus. They use a
richer state-of-the-env spec with:

| Field | Meaning |
|---|---|
| `speed_mean_scale` | Multiplier on per-segment mean travel speed |
| `sigma` | Speed-sampling stddev (higher = more chaotic traffic) |
| `speed_cap` | Hard speed ceiling (m/s) |
| `od_global_mult` | Global passenger arrival multiplier |
| `station_od_overrides` | Per-station multiplier, e.g. `{"X06": 50.0}` |
| `affected_routes` | Which road segments the perturbation hits |

### mega_event (大型活动 — 演唱会 / 赛事 / 跨年)
```python
{
    "speed_mean_scale": 0.5,
    "sigma": 2.0,
    "speed_cap": 10,
    "od_global_mult": 3.0,
    "station_od_overrides": {
        "X05": 30.0,  # 活动场馆站
        "X06": 50.0,  # 主入口站 (3 → 150 pax/min)
        "X07": 20.0,  # 次入口站
        "X04": 10.0,  # 辐射站
        "X08": 10.0,
    },
    "_expected_od_range": [10, 50],
}
```

### holiday_rush (春运 / 黄金周 / 元旦)
```python
{
    "speed_mean_scale": 0.7,
    "sigma": 2.5,
    "speed_cap": 12,
    "od_global_mult": 10.0,  # uniform 10× across network
    "station_od_overrides": {
        "X03": 15.0,  # 商圈
        "X05": 15.0,  # 交通枢纽
        "X10": 12.0,  # 住宅区
    },
    "_expected_od_range": [10, 15],
}
```

### emergency_evacuation (地震预警 / 洪水 / 火灾)
```python
{
    "speed_mean_scale": 0.2,  # roads almost gridlocked
    "sigma": 5.0,              # extreme variance (some roads fully blocked)
    "speed_cap": 5,
    "od_global_mult": 8.0,
    "station_od_overrides": {
        "X06": 40.0,  # 疏散核心站
        "X07": 35.0,
        "X05": 25.0,
    },
    "_expected_od_range": [8, 40],
}
```

### ood_parametric (参数化扫描, for bound-vs-actual plot)
```python
{
    "speed_mean_scale": 1.0,
    "sigma": 1.5,
    "speed_cap": 15,
    "od_global_mult": 1.0,  # overridden at runtime via set_ood_multiplier()
    "station_od_overrides": {},
    "_expected_od_range": [1, 100],
}
```
This is the closest analogue to the current `od_mult` sweep.

### ood_dual_extreme (速度+客流双极端)
```python
{
    "speed_mean_scale": 0.2,
    "sigma": 4.0,
    "speed_cap": 5,
    "od_global_mult": 20.0,
    "station_od_overrides": {
        "X05": 40.0,
        "X06": 40.0,
    },
    "_expected_od_range": [20, 40],
}
```

## What MultiLineEnv currently cannot express

| Legacy feature | MultiLineEnv equivalent | Gap |
|---|---|---|
| `od_global_mult` | `od_mult` | ✓ covered |
| `station_od_overrides` | — | ✗ no per-station control |
| `speed_mean_scale` | — | ✗ no dynamic speed modulation |
| `speed_cap` | — | ✗ no speed cap API |
| `sigma` | — | ✗ no dynamic variance control |
| `affected_routes` | — | ✗ no segment-level targeting |

So option (B1) — multi-burst schedule — adds one new lever
(time-varying `od_mult`) to the set, but cannot express
`mega_event`-style scenarios.

## Migration plan if we ever move to (B2)

If the paper or a future experiment needs the legacy OOD scenarios on
MultiLineEnv, the minimal port is:

1. **Per-station OD override**: `env_bus._pax_flat_rates` is a flat
   `np.ndarray` of rates indexed by a flattened
   `(direction × station)` pair. Need to build a station-name → index
   map during env construction, then apply overrides at
   `_batch_passenger_arrival` time.
2. **Dynamic speed scaling**: segment speeds are sampled per-trip from
   the calibrated data. Add a multiplicative
   `env.speed_scale` field (and corresponding `sigma_scale`,
   `speed_cap`) consulted at sample time.
3. **API**: a single entry point
   `env.set_ood_mode(name_or_dict)` that accepts either a named legacy
   mode or a dict in the same shape.

Estimated size: ~100–200 lines in `offline-sumo/env/sim_core/sim.py`
plus a companion `mode_profiles.py` in the same directory. Not in
scope for the current CS-BAPR paper submission.

## Pointers

* Legacy definitions: `bapr_reference/mode_profiles.py`
* Legacy usage: `bapr_reference/sac_ensemble_baseline_stress.py`,
  `bapr_reference/sac_ensemble_escp_stress.py`,
  `bapr_reference/sac_ensemble_bapr.py`
* Current MultiLineEnv OOD implementation: `offline-sumo/env/sim_core/sim.py`
  (search for `od_mult`, `set_ood_burst`, `clear_ood_burst`)
* Current eval entry: `scripts/test_multiline_convergence.py`
  (`evaluate_ood`, `evaluate_abrupt`)
