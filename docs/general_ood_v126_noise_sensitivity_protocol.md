# v126 frozen noise sensitivity

Use final beta=-2 policies from all three v123 300k sources and v125 350k
controls. No training, optimizer updates, readout refit, reward or physics changes.
Load native final snapshots, not replay checkpoints or best periodic models.

CPU evaluation: reset seeds 150000--150004; noise scales 0/0.25/0.5/1 before
tanh; legacy noise seeds 12400--12402 and fresh seeds 12600--12602, reported
separately. Generate one CPU float32 (1,3) Gaussian draw per step with a local
torch.Generator; this preserves legacy CPU draw sequences at scale=1. CUDA uses
the same explicit draws, not its own RNG. Scale=0 is evaluated once per reset.
Keep episode-local context reset and the checkpoint's long-run context normalizer;
restore context after every episode. Native Hopper-v4 time limit: 1000 steps.

Three CPU tasks evaluate six models / 570 episodes. A fourth CUDA task repeats
seed2's two models / 190 episodes, also comparing CPU/CUDA actor mean/std/action
on identical augmented states and noise every 25 steps. Total: 760 episodes,
at most 760000 evaluation interactions. Noise repeats/devices are not training
seeds. Save scalar summaries, initial normalizers and the final three-step tails,
not complete trajectories. Resume from per-task progress JSON.

Native CPU and CUDA seed2 smokes cover both ages, all scales and both noise
groups using one reset, one noise seed/group and a 64-step cutoff (28 episodes
total). Use the frozen v125 code; do not alter training modules. Checkpoint
availability and simulator/runtime parity currently make node007 eligible;
scheduler chooses resources without node/GPU preferences. No Windows nodes.

Decision: use fresh-noise replication and CPU/CUDA common-input/closed-loop
comparisons to distinguish a noise-seed/device effect from loss of robustness.
Noise-scale response is diagnostic, not permission to select a favorable inference
mode for the paper or proof that entropy caused the failure. Choose a subsequent
single-factor training intervention only after these results. JSON only is synced.
