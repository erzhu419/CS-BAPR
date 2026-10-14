# v146 frozen pulse/recovery diagnosis

Four CPU cases: ID policies from v144 and gravity1.5 policies from v145,
seeds3/4, only their predeclared final600k snapshots. No new training, optimizer
updates, replay loads, context or symbolic constraints. ZIPs remain on servers.

Each policy starts in its own gravity1.0/1.5. After100 or300 completed steps,
apply the unchanged gravity multiplier2.6591479484724942 for10/25/50 steps,
or permanently. Include a no-pulse control: nine conditions/policy. Same five
resets154000--154004 and deterministic/three common noises137000--137002,
1000-step cap:720 episodes. Only gravity changes; native rewards, motor limits,
mass, friction and healthy termination remain unchanged. Switching does not
reset state, elapsed steps, policy or noise. All names stay out of policy inputs.

Retain pre-intervention state/return markers and require paired-prefix equality
against the same-run no-pulse control. Count all outcomes, including failure
before exposure. Recovery requires remaining healthy for100 full steps after
gravity returns to baseline. A fall exactly on the recovery horizon is failure.
Permanent pulses have no recovery horizon. Report exposure and recovery
denominators separately; do not filter unexposed episodes from overall returns.

CPU smokes: ID seed3 and gravity1.5 seed4; one reset/two modes, no-pulse,
after100/pulse10 and after100/permanent;256-step cap. Verify native source,
physics events, no CUDA initialization, zero updates and paired prefixes.
Four formal tasks eligible on all three Linux nodes; no preferences/pins/Windows.
Only scalar/marker JSON locally; lightweight ZIP staging is server-to-server.

## Interpretation
Find recoverable exposure durations while retaining extreme amplitude and a
permanent exposure control. This is frozen-controller diagnosis, not learned
adaptation or a CS-BAPR mechanism comparison. Original seed-specific failures
and root-filesystem failures remain in the v145 evidence record.
