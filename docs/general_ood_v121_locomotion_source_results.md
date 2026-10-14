# v121 results

All three training seeds completed 100000 interactions and 99745 optimizer updates.
Tasks t141773/t141774/t141775 exited zero in 40.3--42.2 minutes. Only 76426 bytes
of scalar results were synchronized; checkpoints remain native.

| Training seed | Final velocity (m/s) | Episode length | Return | Fraction abs velocity <0.1 |
|---|---:|---:|---:|---:|
| 0 | -0.012407 | 1000 | 987.205 | 0.9822 |
| 1 | 0.005441 | 1000 | 1003.721 | 0.9884 |
| 2 | 0.033311 | 1000 | 1032.569 | 0.8900 |

Each row averages five deterministic final-checkpoint episodes on the same evaluation
seeds. Across three independent training seeds, mean velocity is 0.008782 m/s;
all 15 episodes survive to the time limit, but there is no useful locomotion source.
Periodic evaluations also show near-standing behavior, not a late successful phase.

Normal-only training did not solve the source problem. Final 5000-update blocks
have mean Q around 21515--23877, Q loss 0.47--1.54 million, alpha 3.26--23.31,
and reward-normalization standard deviation 0.187--0.677. These are scale diagnostics,
not causal explanations. Decompose the existing target L1 uplift before a single-factor
source-training ablation. Do not rebuild readouts or change manuscript result tables yet.

The native frozen-target probe completed with zero optimizer or environment steps.
Across full-replay and last-10000 pools, discounted L1 target uplift is 269--430
per step, versus normalized immediate reward 1.63--5.44 (ratios 49--263).
The probe reconstructs the actual critic loss. This state-independent positive
continuation bonus is a candidate explanation for excessive survival preference in
a task with fall termination; its causal role is not established by endpoint arithmetic.

Next: v122 trains fresh seeds 0/1/2 for the same 100000 steps, changing only
`weight_reg=0.01 -> 0.0`. Preserve actor beta=-2 and every remaining hyperparameter,
evaluation seed and simulator version. Pair with completed v121 controls; no source
or readout is adopted before locomotion and survival are evaluated together.

Summary: `results/general_ood_v121_locomotion_source/summary.json`.
Target probe: `results/general_ood_v122_target_l1/target_probe/target_scale.json`.
