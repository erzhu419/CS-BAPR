# v83: Whole-seed prediction-error covariance

Keep the v78 local-linear mean, alpha 0.064, 64-neighbor rule, global forward coefficients and input/target scaling fixed. Each reference row's calibration error is its known-mode residual minus the v78 mean predicted from all other reference seeds. Fit zero-centered OAS second moments of these errors in the original 64-neighbor input geometry, with the existing relative covariance ridge 0.001. Mix with raw-residual covariance using one shared raw weight.

Reuse the v82-verified provenance of all 23,000 reference rows. For each of six outer validation seeds, remove every coupled stream and mode label of that seed first. Within the remaining reference bank, cross-fit errors by removing each error-source seed in turn. Thus neither outer validation labels nor the error-source seed enter that error sample's mean fit. Preserve all remaining rows in the error library; no reference thinning or neighborhood rule change.

Use 512 evenly spaced validation rows per outer seed with all five mode labels: 3,072 queries / 15,360 labels. Freeze the global raw weight using the existing bounded Gaussian-NLL optimizer plus endpoints, with equal seed/mode weighting, before loading evaluation outcomes or normal-null traces. Cache completed outer folds on the server. Rebuild the final 23,000-row error library with all reference seeds, each error predicted while excluding its own seed, and restore the full bank for deployment means.

Native checks cover mean/error invariance, raw-density equivalence to v78, pure error second moments and physical density units. If selection returns raw weight 1, retain v78 without duplicate replay. Otherwise replay both pure cross-seed covariance and the selected mixture on the same 65 paths, using identical 12,000-transition normal-null populations and unchanged recursive rules/windows. If weight is 0, these are identical and only the pure condition is run. Retain the six v78/v81 controls as reused statistics.

Judge early/tail accuracy, signed true-mode margins, normal risk and change-minus-no-change contrasts together. No fresh seeds, environment steps, actor updates or paper-table changes in this round. Source code and scalar JSON/logs may be synchronized; error arrays, fold caches and reference banks stay on the server.

## Limitations
Cross-fitting uses smaller mean-fitting banks than full-bank deployment, so errors can overstate deployment uncertainty. Reference validation covers known modes and already-collected trajectories; development replay is not independent confirmation or general RL return evidence.
