# v126 results and next step

All four tasks t142278--t142281 exited zero and automatically synced scalar
JSON (5596933 bytes). Batch wall time: 4.60 minutes. Protocol validation passed:
760 episodes / 423893 evaluation interactions, zero training/optimizer updates.
Source checkpoints stayed native and unchanged.

CPU mean episode lengths below; nonzero-noise entries are legacy/fresh groups,
each with 15 episodes. Zero noise has five episodes.

| Seed | Age | Noise 0 | Noise 0.25 | Noise 0.5 | Noise 1 |
|---|---:|---:|---:|---:|---:|
| 0 | 300k | 147.0 | 146.8 / 146.7 | 146.7 / 146.3 | 146.1 / 145.7 |
| 0 | 350k | 148.8 | 148.9 / 148.9 | 148.9 / 149.1 | 149.3 / 149.8 |
| 1 | 300k | 440.8 | 520.2 / 714.7 | 403.4 / 736.8 | 719.5 / 871.6 |
| 1 | 350k | 1000.0 | 1000 / 1000 | 1000 / 1000 | 759.1 / 397.5 |
| 2 | 300k | 1000.0 | 1000 / 1000 | 959.0 / 1000 | 1000 / 1000 |
| 2 | 350k | 1000.0 | 719.8 / 526.9 | 191.8 / 110.7 | 110.1 / 107.5 |

Seed2's 350k degradation replicates with fresh noise, not just the old noise
set. Its falls at scales 0.25/0.5/1 were 5+8 / 14+15 / 15+15, all angle
terminations. Seed1 at 350k stayed upright in all 60 quarter/half-noise episodes,
but full noise caused 4/15 legacy and 10/15 fresh falls, also angle terminations.
Seed0 fell in every episode, including zero noise: scaling exploration noise
cannot repair its existing mean-policy failure.

CPU/CUDA seed2 comparisons had identical lengths and fall flags in all 190
paired episodes. Maximum same-input action difference was 1.073e-6; maximum
paired return difference was 1.1242 at 300k and 0.4387 at 350k. Device differences
do not explain this observed collapse. Seed2 final alpha remained about 0.07;
deterministic-rollout pre-tanh std was 0.1643/0.1644 at 300k/350k, while full-noise
rollout std was about 0.1577/0.1881. Visited states differ, so do not attribute
failure to a globally inflated std head from those rollout means.

Next: v127 frozen component attribution, submitted; see
`experiments/general_ood_v127_component_attribution_20261011/dispatch_manifest.json`.
For all
three seeds, cross the 300k/350k mean function, std function and initial context
normalizer (eight sampled combinations). Compute mean and std from their full
source actors on the same current augmented state; do not transplant just the
output-layer weights. Use v126's common resets and both noise groups at scale=1,
plus four mean/normalizer deterministic combinations. This is 780 episodes with
zero updates. Retain both coherent-source baselines and every crossed outcome.
Use this controlled decomposition to select a targeted mean-policy, variance or
context update; do not yet expand training seeds or blindly lower entropy.

## Limitations

The attribution experiment is pending. Current results show local loss of noise
robustness after training, not its component cause or OOD superiority. Noise
response is not uniformly monotonic (seed2 300k had one half-noise fall but no
full-noise falls). Inference repeats/devices are not independent training seeds;
future crossed policies are diagnostics, not an adopted algorithm. Paper tables
and earlier negative results remain unchanged. Summary:
`results/general_ood_v126_noise_sensitivity/summary.json`.
