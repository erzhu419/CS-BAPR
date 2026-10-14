# v101: do not replace the incumbent suffix recipe

Task t140802 completed successfully in 2h17m41s: 95 paths, 570000 replayed transitions. The equal-budget comparator, trajectory order, inherited controls and calibration protocol match. Only scalar JSON was synchronized.

V95 updating mode accuracy drops from 68.117% to 64.141%; nonpositive mean posterior-margin windows increase from 7 to 9. Seed135190 gains 2.305 percentage points, but seed135191 loses 10.250 and seed135192 loses 3.984. All three covariance conditions regress on this population. True-head prediction error improves from 0.162557 to 0.156286, which does not establish improved mode discrimination.

The six previously unsupported windows improve accuracy from 14.438% to 28.563%, but their mean posterior margin worsens from -4.381 to -6.243. Compound seed135192 tail crosses to positive mean margin while accuracy falls; three other windows newly cross to negative margin. Historical v62 slippery-tail accuracy also falls, 32.813% to 29.688%. V95 updating maximum normal-window mean risk rises from 0.004383 to 0.005172, although v95 frozen normal risk improves.

Retain v98; do not select age-balanced. Next: cross incumbent/age-balanced means and covariances on the three new failures with normal/frozen controls. Their instantaneous density margins also regress, so posterior history alone is insufficient to explain the outcome. Follow-up v102 is dispatched; execution state is recorded in experiments/general_ood_v102_failure_attribution_20261010/dispatch_manifest.json. No actor integration, seed expansion or paper-table changes.

## Limitations

One source policy, known simulator modes and inspected development trajectories. Accuracy and mean-margin signs are different metrics; normal mean risk is not a false-positive rate. No independent confirmation, causal mean/covariance decomposition or RL-return gain is established by this replay.
