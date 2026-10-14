# v134 frozen source robustness

Evaluate all four predeclared v133 final300k actors, seeds3/4 x MSE/2xHuber10.
No training, readout refit, reward/physics changes or checkpoint selection.
Keep the frozen v133/v132 core and v126 common-noise CPU evaluator.

Per actor: five deterministic anchors150000--150004,20 new deterministic
resets151000--151019, then those20 resets with noise seeds134000--134002 at
native noise scale1. Each episode stops on termination/truncation or1000 steps.
85 episodes per actor /340 total, at most340000 evaluation interactions,
zero new training interactions /zero optimizer updates. Reset/noise repeats
are not additional independent training seeds.

Load only lightweight final.pt snapshots on servers, no replay. Before every
episode restore the trained context normalizer, reset the short EMA, then
update context from actual transitions. All actor/critic parameters stay frozen.
Record CPU anchor outcomes alongside v133 GPU anchors; long trajectories may
differ numerically and are not required to reproduce exactly.

Report survival, return, forward velocity and physical ending causes separately
for anchor/deterministic-new-reset/sampled-new-reset groups. Preserve both Huber
outcomes. Decide normal sampled-policy readiness before OOD/baseline expansion.

Tests cover fixed-noise/frozen-context evaluation and episode-level resume.
Native smoke: one model per loss, three cells each with64-step cutoff. Formal
jobs use the full85-cell protocol, not the smoke cutoff. JSON progress supports
resume. Native CPU execution must not initialize CUDA.

Use node007/jtl110gpu/jtl110gpu2 with no node/GPU preference, no Windows nodes;
CPU2 /RAM2048MB /VRAM0 per task. New-node runtimes live on data disks. Final
snapshots stage server-to-server; only small robustness.json syncs locally.
Historical sources/results and paper tables stay unchanged.
