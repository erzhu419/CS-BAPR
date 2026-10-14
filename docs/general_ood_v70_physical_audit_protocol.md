# v70: physical separation on fresh shifted paths

Freeze the v69 reference and the seed1 source actor (99,745 updates). Run seeds
135070/135071, each with normal, high-gravity and compound branches: 6,000 steps,
change at 3,800, native TimeLimit 1,000, stochastic actions, fixed beta -2.
Context still observes real transitions. No optimizer updates or new RL replay.

For every real step, restore the same full MuJoCo integration state, RNG and
TimeLimit age into five fixed-profile shadows. The true-mode shadow must match
observations, unclipped positions/velocities, reward and termination exactly.
Compare observable template separation against frozen v69 model approximation
and mode likelihoods, especially first-post50 and last64. Normal prefixes must
be identical within seed. No new threshold fitting or online mode labels.

Budget: 36,000 real + 180,000 shadow = 216,000 simulator steps, zero updates.
Full states, traces and resume checkpoints remain server-side; pull scalar JSON
and logs only. Use eligible Linux nodes with those server-only inputs, with no
GPU/machine preference and no Windows.

If exact replay fails, repair state/profile restoration before interpreting
results. If templates coincide observationally, measure the loss of information
before proposing a larger reference. If physical separation survives but model
error dominates, the next candidate should address shifted-state model coverage
or approximation. This round does not fit that candidate.

## Limitations

The physical oracle uses privileged state and known profiles. Frozen-policy
paths are new diagnostics, not reproductions of old updating-policy paths, and
do not establish closed-loop control gains or extreme-OOD robustness.
