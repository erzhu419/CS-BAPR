# v75 Support Attribution

Task `t137083` completed on node007 in 60.69s, exit code 0. Three tests passed.
The diagnostic read the 55 existing v74 evaluation paths without simulation,
optimizer updates or reference fitting. Only the 188,249-byte scalar JSON was
retrieved: `results/general_ood_v75_support/support/support_attribution.json`.

## Findings

No velocity saturation was observed in reference inputs or the first 50
post-switch / last 64 transitions' inputs and targets on any evaluated path.
The v74 slippery extension supplies 29.58% of fresh frozen changed-path early
neighbors, but only 0.14% for v62 updating paths and 0.49% for v61 updating paths.

The same-seed v62 normal controls isolate the behavior shift: mean 64-neighbor
radius increases from 0.5045 (frozen) to 1.0777 (updating). Neighbor state RMS
increases from 0.1074 to 0.2298; action RMS from 0.1492 to 0.3218. These are
distances in the unchanged input-standardization coordinates, not error rates.

## Next Step

Target updating-policy state/action coverage. Compare frozen and updating
normal/slippery references using the same reference seeds, profile mixture,
time sampling and 4,400 added labels per mode. Retain the original v72 bank and
v74 slippery control. Freeze forward coefficients, scales and temporal rules
before two new updating-path test seeds across all five modes. Measure normal
risk and heavy/gravity early regressions alongside changed-mode identification.

## Limitations

These are development diagnostics, not a new RL return comparison. Absence of
clipping applies to the inspected windows, not every historical transition.
Neighbor distance and membership identify a coverage gap but do not prove that
coverage alone explains every error. No observation or controller change is justified yet.
