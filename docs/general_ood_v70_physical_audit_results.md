# v70 result: learned likelihood fails despite physical separation

t136913 completed on node007 CPU, exit 0, 162.31 seconds. All 36,000 true-mode
replays matched observations, unclipped physical state, reward and termination
exactly. Budget: 36,000 actual + 180,000 shadow steps; zero optimizer updates.
Only the 179,596-byte scalar JSON was pulled; raw states/checkpoints stay remote.

All four OOD first-post50 and last64 windows had unique observable true templates
on every transition. No input velocity clipping occurred. Normal mean risk was
at most 0.01731 across first800/full/last1000/last64. Last64 results:

| Seed | Mode | Median physical gap | Median model error | Median error/gap | 16-step mode accuracy |
| --- | --- | --- | --- | --- | --- |
| 135070 | high_grav | 0.09338 | 0.12543 | 1.306 | 0% |
| 135071 | high_grav | 0.10059 | 0.13019 | 1.264 | 0% |
| 135070 | compound | 0.13051 | 0.37619 | 3.012 | 0% |
| 135071 | compound | 0.12578 | 0.38654 | 3.091 | 0% |

Gap/error RMS use the unchanged v69 target scales; ratios are medians of
per-transition ratios. All four tails were classified as normal. Early paired
risk contrasts were positive 4/4, tail contrasts only 2/4. Keep v69 HOLD.
The observed failure is model/likelihood generalization on shifted paths, not
missing physical information in these windows. Coverage versus quadratic-model
approximation remains unresolved. Next: equal-label-count normal-only versus
shifted-state reference expansion on new seeds, with unchanged model capacity,
readout window and threshold rules; never fit the v70 audit seeds.

## Limitations

One source training seed, frozen weights with causal context updates, mild known
profiles and privileged simulator templates. This is neither old updating-path
reproduction nor deployable belief/closed-loop performance confirmation.
