# v58 matched closed-loop pilot

Four arms: fixed-minus2, fixed-minus2p3, pooled-beta and phase-beta. New
continuation seeds116000/116001, each6000 interactions/updates per arm; total48000.
Same v40 seed1 source networks/optimizers/counters, v52 fit-only replay6144 and
context100000, v53 dynamics and v57 formal pooled/phase references. Frozen
moments, loss code, reward normalization and beta=-2-coefficient (scale1).
Feedback changes only the inherited actor beta slot; static controls use0/.3.

Train stochastic actions. A seeded permutation of the four original profiles
runs at steps800/1800/2800/3800, with normal at0 and4800: each changed profile
gets1000 steps and normal gets2000. The schedule and reset seeds match all arms
within a continuation seed; trajectories may diverge after the first update.
Maintain residual/departure history across real episode resets; phase selection
uses no OOD label. Monitor readout/actor beta, losses and rewards by actual mode.

Evaluate a restored final checkpoint, deterministic actions, zero updates and
training context restored before each episode. Two paired episodes each for
ID, heavy, slippery, high_grav, compound and structured50x Markov (200-step
switches, probability1); evaluation bases120000/121000 plus profile offsets.
No checkpoint selection, beta tuning or new reference fitting from outcomes.

Save full replay/RNG/online readout/coefficient and native state every500 steps.
Only scalar JSON/logs return. Review paired returns, ID retention and normal
readout drift before expanding; no automatic performance PASS from detector
signs. Two continuation seeds share one trained source and are only a pilot.
