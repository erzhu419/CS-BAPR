# v146 frozen pulse/recovery results

All four CPU tasks t151854--t151857 completed exit0:720 episodes /300663
steps,640 paired prefixes verified, zero training/updates/CUDA initialization.
2355515 JSON bytes pulled; no policy/replay downloads. Runs took51--82 seconds.

The unchanged2.659 gravity peak lasts10 steps after100 completed steps:

| Source600k policy | Sampled length /15 | Healthy100-step recovery | Full1000-step finish |
|---|---:|---:|---:|
| ID seed3 | 874.87 | 15/15 | 12/15 |
| ID seed4 | 250.40 | 15/15 | 0/15 |
| Gravity1.5 seed3 | 202.00 | 3/15 | 0/15 |
| Gravity1.5 seed4 | 250.53 | 15/15 | 0/15 |

Short-horizon recovery is not long-term stability. Later onset changes outcomes:
ID seed3 after300/pulse10 finishes15/15, while ID seed4 finishes5/15. At25 steps
after100, neither ID policy recovers; the25/50-step and permanent controls remain
strong failure cases overall. Retain onset/seed separation and unexposed failures.
Next plan: matched fresh-seed no-pulse versus randomized brief-pulse SAC training,
with original extreme amplitude/durations held out. No further tasks submitted.

## Limitations
Frozen reactive policies, not CS-BAPR adaptation. Two training seeds; repeated
resets/noise are not independent seeds. The horizon metric never substitutes
for full survival, and permanent exposure is a control, not an assumed outcome.
