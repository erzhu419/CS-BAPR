# v148 known-pulse results

t152031--t152034 exited0 on node007 CPU.400 frozen episodes/305,382 steps,
320 same-run prefixes verified, zero training/updates/replay/GPU initialization.
Only1,419,009 bytes of formal JSON were pulled; final600k ZIPs stayed on servers.

Mean length / healthy finishes,15 sampled episodes per cell; pulse duration10.

| Policy | Seed | No pulse | 2.0x after100 | 2.0x after300 | 2.659x after100 |
|---|---:|---:|---:|---:|---:|
| No-pulse training | 5 | 1000/15 | 1000/15 | 1000/15 | 191/0 |
| Randomized recovery | 5 | 509/0 | 470/0 | 491/0 | 264/0 |
| No-pulse training | 6 | 1000/15 | 1000/15 | 1000/15 | 305/1 |
| Randomized recovery | 6 | 1000/15 | 1000/15 | 742/6 | 1000/15 |

All15 sampled episodes pass100-step recovery for every2.0x condition, yet
seed5 randomized policies all subsequently fall. Within-training-support
recovery and normal-policy stability are therefore insufficient, not just
extreme-amplitude extrapolation. The seed6 early-extreme gain remains real
but does not extend reliably to later pulses. Next: matched warm recovery
continuation from stable v147 no-pulse600k states (v149 plan, not dispatched).

## Known Issues

First native smokes rejected cross-host roundoff under exact equality
(observations identical; inspected state/return differences about1e-14).
Controls were recomputed within each run; v147 evidence remains valid.
Two training seeds remain exploratory, not a CS-BAPR superiority result.
