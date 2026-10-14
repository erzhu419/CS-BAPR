# v119 actor-feedback results

All 12 tasks t141665--t141676 completed with exit 0. Frozen metadata/first
transitions match: 60 paths, 360000 updates, 307265 evaluation steps; scalar sync 1175025 bytes.

Equal-weight means over three environment seeds and four changed modes:

| Metric | Matched | v98 | Fixed -2 | Fixed -3 |
| --- | ---: | ---: | ---: | ---: |
| Changed post-switch reward/step (primary) | 1.03862 | 1.04258 | 1.03020 | 1.03339 |
| Changed final deterministic episode return | 967.49 | 963.36 | 990.03 | 1014.79 |
| Normal-only final episode return | 947.45 | 924.16 | 1046.48 | 1041.71 |
| Normal return after changed training | 1004.10 | 988.97 | 991.96 | 1035.05 |

Matched-minus-v98 primary difference -0.00396, paired seed-level 95% t interval
[-0.04921,0.04130]; versus fixed-2 +0.00842 [-0.01113,0.02797]; versus fixed-3
+0.00523 [-0.00460,0.01507]. No overall advantage; final-return intervals also cross zero.

Matched normal returns by seed: 777.05,1030.49,1034.80; v98:1046.12,687.80,1038.58.
Both adaptive arms have an adverse normal seed; both constants remain above1033.

Feedback is active: candidate changed risks0.946--0.985, normal0.0012--0.0023;
v98 changed0.817--0.976. Improved recognition does not establish useful scalar
feedback; near-saturated changed risk largely approaches fixed-minus3.

Default healthy reward1 and action cost at most0.003 imply matched mean forward
velocity only0.039--0.042. Next: v120 native reward decomposition, unchanged-source
evaluation and common-input uncertainty-gradient diagnosis. No retraining,
classifier tuning, seed expansion or adoption; paper tables unchanged.

The completed v120 diagnosis confirmed the low-velocity source and nonzero
uncertainty gradients; see `general_ood_v120_mechanism_results.md` for the next
source-training plan. v119 is retained as a conditional negative result.

## Limitations

Three environment seeds, one source, mild Hopper modes, short budget and three
evaluation episodes. Divergent post-feedback trajectories and different reference
budgets. Exploratory per-mode intervals are not multiplicity-adjusted. Only final
compound checkpoints survive latest.pt overwrites; no historical gradient claim.
