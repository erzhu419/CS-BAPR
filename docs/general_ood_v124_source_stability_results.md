# v124 results and next step

All three CPU tasks t142131/t142132/t142133 exited zero. Six frozen models,
120 episodes and 60784 evaluation interactions; zero optimizer updates and no
readout refit. Scalar JSON transfer: 580756 bytes. Source checkpoints stayed native.

| Seed | Age | Deterministic length (falls / 5) | Sampled length (falls / 15) |
|---|---:|---:|---:|
| 0 | 100k | 102.2 (5) | 102.1 (15) |
| 0 | 300k | 147.0 (5) | 146.1 (15) |
| 1 | 100k | 249.2 (5) | 254.2 (15) |
| 1 | 300k | 440.8 (5) | 719.5 (7) |
| 2 | 100k | 1000.0 (0) | 850.7 (4) |
| 2 | 300k | 1000.0 (0) | 1000.0 (0) |

Sampling helps seed1 at 300k, harms seed2 at 100k, and cannot rescue seed0.
Fall conditions were height and/or angle, with no state-range violation. All
deterministic seed0/1 falls crossed the height boundary; seed0 at 300k also
crossed the angle boundary in 4/5 episodes.

At 300k, seed0 full-replay risk-gradient/non-risk-gradient ratios were
1.517--1.805, and total/non-risk gradient cosines were -0.236 to -0.053.
The risk term can reverse the raw actor-gradient direction despite its modest
scalar value. Seed1 recent-replay ratios were 1.513--1.687. Stable seed2 also
had substantial risk gradients (recent ratios 1.018--1.494).

Next: v125 forks every 300k source into actor-beta=-2 and actor-beta=0 arms,
each adding 50000 interactions/updates (six jobs, 300000 new interactions).
Keep critics, targets, entropy tuning, behavior cloning, replay, optimizer state,
reward, seeds and final-only selection unchanged. Evaluate both deterministic
and sampled actions with a common protocol. The beta=-2 arm must also continue;
the frozen 300k source is not a matched-budget control. This plan is implemented
in v125; see `experiments/general_ood_v125_actor_beta_20261011/dispatch_manifest.json`
for its task state.

## Limitations

These are raw frozen gradients, not historical clipped Adam updates or causal
proof of harmful pessimism. Physical flags identify termination conditions, not
the underlying dynamics causing a fall. CPU diagnostics and original GPU
evaluations remain distinct protocols; noise repeats are not independent training
seeds. No algorithm adoption, OOD superiority claim or manuscript-table change.

Summary: `results/general_ood_v124_source_stability/summary.json`.
