# RandomLoad v6: non-learning pilot

The first frozen environment candidate passes all predeclared control checks on paired seeds 40000--40099. No learning algorithm was run or ranked during environment selection. The 47-test configuration/environment/reference/resume suite passes.

| Severity | Zero Cost | Force-Blind Feedback Cost | Compensated Feedback Cost |
| --- | ---: | ---: | ---: |
| 1 | 90.6728 | 0.1913 | 0.1912 |
| 2 | 91.1727 | 0.1919 | 0.1915 |
| 5 | 116.7532 | 0.1990 | 0.1951 |
| 10 | 250.1957 | 0.2314 | 0.2119 |
| 20 | 931.0179 | 0.3865 | 0.2938 |
| 50 | 7,841.5814 | 2.0673 | 1.1372 |

Compensated feedback never exceeds the saturation threshold |action|>0.95 over the pilot. The unchanged action authority handles the disturbance; observed current-force compensation improves 50x cost over force-blind feedback. Extreme randomness is ongoing rather than confined to the reset. This pilot establishes a usable control diagnostic, not any CS-BAPR advantage or optimality of the reference.

At 50x, measured storm occupancy is 35.175% versus 5.765% in ID, and mean episode force RMS is 0.4561 versus 0.0041 (about 112x). The disjoint-seed-996 six-episode CUDA preflight completes with exit code 0, 1,200 learning interaction steps and 201 updates.

Next: five v5 presets, seeds 0/1/2, 500 episodes each, under the frozen `docs/random_load_v6_protocol.md`. No loss-weight, efficiency or legacy-environment changes. Full pilot evidence is `results/random_load_v6/environment_pilot.json`.
