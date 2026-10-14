# v132 source stability pilot

Fresh matched Hopper-v4 sources, seeds3/4, two arms: `mse-control` and
`huber10`. Each receives100000 interactions /99745 optimizer updates;
total400000 /398980. No old replay or checkpoint continuation.

Both arms reset episodes/context on terminated-or-truncated, but store only
physical termination as the Bellman mask. Timeout next states retain bootstrap.
The only between-arm change is critic TD loss: MSE versus2xHuber(delta10),
identical quadratic loss/gradient below10 and residual derivative capped20
above10. The existing ensemble disagreement penalty remains unchanged.
Delta is fixed in normalized critic-target units, not raw reward units.

Keep beta-2, target L1 weight0, native reward/physics/1000-step limit, context,
reward EMA, automatic alpha, minimum-Q targets, gradient clipping1 and no
random-action warmup. NAU/SINDy/symbolic penalties remain disabled. Freeze the
v131/v130 core rather than copying the dirty current core. Existing actors,
results and paper tables remain unchanged.

Use the same periodic deterministic evaluations (diagnostic only), final-only
actors and five final reset seeds150000--150004. Record squared TD error and
actual TD objective separately; Huber training loss is not comparable to MSE
as a numerical-stability metric. Seed4 acquisition and seed3 stability are
separate endpoints. This is not an OOD performance claim.

Before formal release: transition-mask/loss/resume tests and native CUDA smoke
for both arms. Scheduler can use node007, jtl110gpu or jtl110gpu2 with no node
or GPU preference and no Windows nodes. Code/checkpoints on data disks on the
two new nodes; full states publish server-to-server to central storage. Only
small scalar `source.json` files sync locally. No local checkpoint/replay pulls.

Next gate: analyze both seed pairs at100k. Extend the same protocol to300k only
if justified by stability and locomotion, before any retention/OOD expansion.
