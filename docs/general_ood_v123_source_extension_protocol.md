# v123: unchanged-source training continuation

Continue all three v122 training seeds 0/1/2 from their native full `latest.pt` at
100000 interactions to 300000 interactions. Add 200000 interactions and optimizer
updates per seed: 600000 new interactions total, plus 300000 reused prefix interactions.
Restore replay, belief/surprise/context, optimizers, reward EMA, RNG, partial episode
and MuJoCo integration state. Keep v122 sources unchanged in their original directories.

No hyperparameter, reward, physics, exploration or algorithm changes: weight_reg=0,
actor beta=-2, context MLP, existing disagreement/BC terms and reward/entropy scaling.
Record a context/RNG-preserving evaluation at the 100000-step handoff, then keep the
v121 evaluation/checkpoint schedule. Evaluate the saved final 300000-step model on
five common seeds 150000--150004. Intermediate evaluations never select checkpoints.
Assess speed jointly with episode length and fall termination; retain unsuccessful seeds.

This is a budget continuation, not a matched-budget comparison against v121's original
L1 coefficient. The completed v122 100000-step single-factor evidence remains separate.
Do not rebuild readouts until source locomotion and survival are sufficiently reliable.

Use the native v122 runtime and let scheduler choose GPUs without preferences;
no Windows. Only scalar JSON is synchronized; checkpoint/replay data remain native.
Smoke restores the real seed0 100000-step checkpoint and performs 512 additional updates,
with shortened diagnostic/evaluation intervals. Formal tasks restart from v122, not smoke.
