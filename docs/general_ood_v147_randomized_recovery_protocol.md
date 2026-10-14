# v147 randomized recovery training

Fresh standard SB3 SAC 2.7.0, seeds 5/6 crossed with `no-pulse` and
`randomized-recovery`; 600,000 interactions and 599,745 updates each.
Settings, native Hopper rewards, 11-dimensional observations, motor limits,
termination and 1,000-step episode cap are unchanged. Snapshots are fixed at
300k/600k, not selected using held-out performance. Core comes from frozen v146.

Each training reset uses `seed*10000 + reset index`. In the randomized arm,
an independent `RandomState(147000 + training seed)` selects half the episodes
in expectation, then draws an onset uniformly from 80 through 320 completed
steps. Gravity is 2.0 for exactly ten executed steps, then returns to 1.0.
Control episodes remain at 1.0. Neither arm receives mode labels or clocks;
there is no state reset at either switch. Early falls before onset are retained.
Counters record assigned/exposed pulses, restored episodes, peak/baseline
steps, healthy 100-step recovery and eventual healthy time limits separately.

Full-state resume preserves model/optimizers, replay, native MuJoCo state,
reset index, monitor state, policy RNGs, private pulse RNG, active gravity,
onset, wrapper clock and coverage counters. A task-local persistent SB3 logger
avoids repeated unused `/tmp/SB3-*` directories.

At each snapshot, frozen CPU evaluation uses unchanged v146 schedules: normal
gravity; original 2.6591479484724942 gravity after 100/300 steps for 10/25/50
steps or permanently. Resets 154000--154004 and action noise seeds
137000--137002 plus deterministic actions are shared, giving 180 episodes
per snapshot and 1,440 total. Paired no-pulse prefixes are checked. Test peak
amplitude and 25/50-step durations are outside training support. Recovery after
100 steps and final 1,000-step survival are distinct outcomes.

Native GPU smokes and local parity/resume tests precede formal dispatch.
Placement is free across available native Linux nodes, with no node/GPU
preference or hard pin. Only scalar JSON/logs return locally; checkpoints and
replay stay on servers, with existing server-to-server publication.

## Scope

This tests recovery-state coverage under unchanged SAC, not a CS-BAPR
mechanism. Seed-level paired arm comparisons are primary; evaluation repeats
are not independent training seeds. A positive result motivates matched
CS-BAPR/BAPR/RE-SAC comparisons; it does not itself establish their advantage.

## Known Issues

jtl110gpu has zero available root-filesystem space and cannot create this
experiment's path on 2026-10-14. This campaign uses node007 and jtl110gpu2;
the latter stores the runtime and checkpoints under `/mnt/data`. No server
cleanup or scheduler-wide placement-policy change was made.
