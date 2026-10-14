# v122 results and next step

All three matched training seeds finished 100000 interactions and 99745 updates;
tasks t141961/t141962/t141964 exited zero in 42.1--45.0 minutes. Scalar JSON: 80694 bytes.

| Seed | Final velocity (m/s) | Length | Return | Fall terminations / 5 |
|---|---:|---:|---:|---:|
| 0 | 1.264787 | 102.2 | 231.437 | 5 |
| 1 | 2.254694 | 249.2 | 810.918 | 5 |
| 2 | 1.706633 | 1000.0 | 2706.121 | 0 |

The single-factor removal of target L1 uplift restored movement and reduced frozen
mean Q from about 21631--24210 to 149--196. Mean velocity increased by 1.733256 m/s
relative to matched v121 controls. Survival deteriorated for seeds 0/1; only seed 2
demonstrated sustained locomotion. Mean return improvement (+241.660) hides return
regressions in two seeds. This is not yet a consistently useful source or an OOD result.

Next: v123 retains every v122 seed and continues each complete 100000-step prefix
to 300000 total steps, with no further hyperparameter change. Seed 1's learning curve
and seed 2's late acquisition motivate the added budget; retain seed 0's plateau too.
Keep the 100000-step sources immutable, select only the final 300000-step snapshots,
and report new versus reused interactions separately. No readout refit or paper-table change.

Summary: `results/general_ood_v122_target_l1/summary.json`.
