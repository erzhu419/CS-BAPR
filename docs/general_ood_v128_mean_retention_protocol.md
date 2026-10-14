# v128 mean-only source retention

From every full v123 final 300k source (seeds 0/1/2), fork unchanged-control and
mean-only-source-regularization continuations for exactly 50k additional steps.
Both resume optimizer, replay, simulator, context and RNG; beta=-2, weight_reg=0,
native normal Hopper-v4 physics/reward, no readout refit. Six paths / 300k new
interactions and updates. Select final 350k only, never the best periodic model.

The treatment adds `alpha.detach() * D_mean` to the existing actor objective;
`D_mean = 0.5 * mean_batch(sum_action(((mu-mu_ref)/std_ref)^2))`. Reference mean
and std come from the frozen complete 300k actor on the same current augmented
replay inputs. Coefficient one for all seeds; no tuning. This is the KL of two
Gaussians sharing frozen reference variance, not the current actor's full KL.
No direct current-std penalty or initial-normalizer swap. Shared trunk updates
may still affect std. Track divergence, added loss and deterministic-action MSE.

Final evaluation is CPU-only with five resets 150000--150004, deterministic
actions plus full-scale shared noise groups: legacy 12400--12402, v126-seen
12600--12602, new heldout 12800--12802, reported separately. Native 1000-step
limit: 50 episodes/path, 300 total. Evaluation restores context and RNG, with
zero updates. Keep all seeds and negative outcomes. Test whether seed2 improves
without losing seed1's progress or worsening seed0; this is not OOD evidence.

Local tests cover control update identity, correct mean-only gradients, frozen
reference, complete continuation/resume, evaluation restoration and merger
rejection of mismatched protocols. Native seed2 smoke: both arms, 512 steps,
four final episodes each at the full 1000-step limit. Dispatch formal jobs only
after both exit zero with finite metrics and a positive treatment penalty.

Use the frozen v127 runtime plus the auxiliary actor-loss hook and v128 scripts.
Native snapshots/replay/runtime are on Linux node007; no checkpoint transfer,
machine/GPU preference or Windows nodes. Scheduler chooses GPUs. Sync only
small mean_retention.json; full resumable and final checkpoints remain native.
