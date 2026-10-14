# v136 matched-warmup context test

Freeze the same v133 MSE final300k policies, training seeds3/4. Restore the
source normalizer and reset EMA per episode. Both inference arms update
context on200 identical ID transitions; held-context then keeps the full
tracker unchanged, while online continues observing real transitions.
An ID hold control separates ordinary context dependence from OOD effects.

Four single abrupt changes affect step201, after200 ID steps, using the
existing easy/heavy/mega/slippery structured profiles at intensity50. No
sensor/actuator/reward noise or reward shaping. The first affected action
still uses identical pre-switch context; online can respond from step202.
Profile labels never enter the policy. Log actual physical multipliers,
post-warmup return/survival, context changes and all early failures.

Five new resets153000--153004, deterministic plus native action-noise
seeds136000--136002,1000-step limit:100 paired cells per training seed,
400 episodes /at most400000 interactions. Two seed shards each evaluate
both arms, checking exact state/context/action/transition/physics equality
through step201 in memory. Only scalar equality evidence is published,
without hashes or trajectory downloads. Resume at completed pair boundaries.

Native smokes use ID and abrupt-mega, one reset, deterministic and sampled,
256-step limit. Verify nonzero held context,200 updates in that arm, exact
prefix equality and actual switch at201 before releasing formal tasks.
CPU2/RAM2048MB/VRAM0, eligible node007/jtl110gpu/jtl110gpu2 without preferences
or Windows. Frozen v135 core; no new training, optimizer updates or replay.
Final snapshots stage server-to-server only; small warmup_context.json
results sync locally. Older environments, actors and paper tables unchanged.
