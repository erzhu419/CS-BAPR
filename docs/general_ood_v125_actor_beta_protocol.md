# v125 matched actor-beta continuation

Fork every full 300k v123 source (training seeds 0/1/2) into beta-minus2 and
beta-zero arms. Only the coefficient on actor ensemble disagreement changes:
`-(Q_mean + beta * Q_std - alpha * log_prob) + beta_bc * BC`.
An experiment-specific agent overrides the beta calculation; the shared default
formula remains unchanged. Critic disagreement regularization, min targets,
weight_reg=0, entropy tuning, BC, context, native reward/physics and all other
configuration fields stay fixed. This is not vanilla SAC.

Each arm receives 50000 additional interactions and updates, ending at 350000
total interactions / 349745 total updates. Six paths add 300000 interactions;
the three unique reused 300k source histories contribute 900000 earlier steps.
Restore replay, optimizers, normalization, RNG, physical state and partial episode;
retain source files and separate output directories. The control also continues.

Select final 350k snapshots only. Common CPU evaluation: reset seeds
150000--150004, deterministic actions and three sampled-policy noise seeds
12400/12401/12402, native 1000-step time limit. Report the two modes separately.
Periodic GPU evaluation is diagnostic and never selects checkpoints. Training
seeds, rather than inference-noise repeats, are the independent comparison units.

Both arms must pass native GPU smoke from seed0's real 300k state (512 new
interactions/updates). Local checks cover original beta=-2 equivalence, first-update
critic/alpha identity across arms, actual policy-loss difference, full-state resume
and cross-arm resume rejection. Source and new checkpoints stay native; scalar
JSON only is synced. Scheduler chooses GPUs with no preference; node007 eligibility
reflects native full states and runtime parity. No Windows nodes.

Decision: test whether removing the actor risk term improves sustained locomotion
without damaging seed2. Keep all seed outcomes. Do not refit OOD readouts or adopt
beta=0 as the method before this matched comparison is complete.
