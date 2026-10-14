# v127 frozen component attribution

Use the final v123 300k and v125 beta=-2 350k policies for seeds 0/1/2. Cross
complete mean functions, complete std functions and initial context normalizers
into eight sampled combinations per seed. Both actors receive the same current
augmented state; no output-layer weight transplant. The selected normalizer
supplies initial count/mean/m2, resets episode-local EMA, then updates context
on actual executed transitions. Source objects are restored after every rollout.

CPU only, native Hopper-v4 reward/physics/time limit unchanged. Resets
150000--150004; shared v126 Gaussian schedules at scale=1 with legacy seeds
12400--12402 and previously seen fresh seeds 12600--12602, reported separately.
Four mean/normalizer deterministic combinations omit the unused std factor.
Three jobs / 780 episodes / at most 780000 evaluation steps; zero optimizer or
training updates, no readout refit. Crossed policies are diagnostic interventions.

All coherent 300k and 350k combinations must reproduce native v126 CPU episodes
(210 formal controls). Fail on length/termination mismatch or material scalar
difference rather than interpreting a broken baseline. A seed2 native smoke uses
one reset and one noise per group, all twelve combinations and the full 1000-step
limit: 20 episodes, including six matched v126 controls. Progress JSON supports
episode-boundary resume; JSON scalar summaries and last-three-step tails only
are synced. No checkpoint/replay download or core training-code change.

Use the frozen v126 runtime and native final models on node007. No machine/GPU
preference or Windows nodes; all jobs request CPU resources only. Report every
combination, source seed and noise group, plus single-component contrasts against
the coherent 350k policy. Use those results to choose the next training change;
do not select a hybrid as the final algorithm or count noise repeats as seeds.
