# v133 budget-only continuation

Continue all four v132 final100k full states: seeds3/4 x MSE/2xHuber10. End at
300000 interactions /299745 updates per model, adding200000 interactions and
updates each (800000 new interactions /800000 updates total). Do not select
only the best seed/arm or restart from the lightweight evaluation actor.

Preserve replay, optimizers, alpha, target networks, reward normalization,
belief/surprise/context, RNG, MuJoCo state and in-progress episode. Keep the
frozen v132 core, corrected terminated-only mask, beta-2, target L1 weight0,
Huber delta10, native rewards/physics/time limit and all other hyperparameters.
NAU/SINDy remain disabled. No new seeds, threshold sweep or retention penalty.

Same diagnostic evaluations every25k (not used for selection); five final-only
reset seeds150000--150004. Analyze each paired seed separately: survival,
return and squared TD error, not Huber objective alone. No OOD evaluation yet.

Before formal launch: handoff/resume tests for both losses, including a source
ending mid-episode or at a real timeout, and two native CUDA continuations
adding512 interactions each. Smoke sources remain unchanged; formal tasks use
the original100k endpoints, not the100512 smoke endpoints.

Scheduler may use node007, jtl110gpu and jtl110gpu2 with no node/GPU preference,
no Windows nodes. New-node runtime/checkpoints stay on data disks. Full states
move server-to-server and publish centrally; only small source JSONs sync
locally. Record source and continuation GPU models; continuation across GPU
models is not a bitwise replay claim. Previous actors/results/paper unchanged.
