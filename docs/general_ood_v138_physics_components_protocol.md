# v138 physical-component learnability

Split v137's structured intensity50 mega profile into mass-only3.6363033113,
gravity-only2.6591479485, and friction-only0.3760603093. Other factors stay1.
Use the existing physical activation code; do not alter inertia, actuators,
sensor/action/reward noise, health termination or rewards. Labels are metadata,
not policy inputs. Component controls isolate the compound failure mechanism.

Fresh MSE policies, seeds3/4,300k each: six runs /1.8M steps /1798470 updates.
Reuse frozen v137 core, v132 driver, identical configuration, initialization,
training reset scheme, bootstrap masks, online context and diagnostic intervals.
Periodic evaluations must use their matching component. Select final, not best.
Restore full native state/RNG/replay for resume; final CPU inference loads no replay.

Retain v137's final evaluation: resets154000--154004, deterministic plus common
native action-noise seeds137000--137002,1000-step limit.120 new episodes. Reuse
v137's ID and compound-mega policies as four matched controls; do not retrain,
select or pull their snapshots. Validate matching configuration, budgets and
evaluation protocol before comparison. This is not a random-regime benchmark.

Before release, test exact single-factor changes, native/wrapped physics parity,
inference factory wiring, untouched v137 defaults and exact interrupted resume.
Three native CUDA smokes, seed3,512 steps /257 updates, cover all components.
CPU2 /RAM4096MB /VRAM1024MB, eligible node007/jtl110gpu/jtl110gpu2, no preferences
or Windows. Snapshots remain or move between Linux servers; only small source.json
and logs pull locally. Historical results and paper tables remain unchanged.

## Limitations
Only two training seeds and this300k training setup. Single-factor success does
not prove compound feasibility, and failure is not a physical-impossibility proof.
