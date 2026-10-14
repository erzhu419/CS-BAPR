# v114 results and next step

Completed: native smoke t141553 and formal t141554, both exit 0. Formal scheduler
wall time was 97.76 s. All 76 original windows / 4,248 transitions reproduced
v113's ten scalar statistics per window. Logit decompositions reconstructed the
full margins. Simulator-center diagnostics covered 74 windows / 4,120 transitions.
Local tests: 4 passed. Only 505,943 bytes of formal scalar JSON returned locally.

Equal-window instantaneous classifier accuracy:

| Population | Fitted centers | Simulator centers |
|---|---:|---:|
| Current four failed updating | 15.50% | 99.50% |
| Current six normal | 99.67% | 100.00% |
| Current four matched frozen | 75.50% | 95.00% |
| Earlier six failures | 54.23% | 70.23% |
| All 28 frozen changed windows | 87.17% | 94.81% |

The current failures are highly sensitive to mean inputs: seed135251/slippery
38% -> 100%; seed135252/slippery 10% -> 98%, high_grav 8% -> 100%, compound
6% -> 100%. Their mean best-other margin changes from -1.485 to +0.672.
With fitted centers, its intercept/linear/squared contributions are
-0.168/+1.136/-2.453. Squared terms are adverse in this decomposition, but this
does not justify deleting them globally.

Correct centers do not fix the scoring rule everywhere: seed135192/high_grav
first50 remains at 28% (nearest-mean control 100%); seed135192/compound first50
falls from 82% to 64%. Across 74 paired windows the classifier improves on 36,
regresses on 11, and is unchanged on 27. The six largest frozen regressions from
v113 all improve with correct centers, including seed135191/slippery 34% -> 98%.

## Next step

v115 should first benchmark a shared-trunk, five-mode residual mean predictor
against the local-linear predictor. Use the existing reference data, frozen
forward model and scales, matched training rows and complete-source holdouts;
no new collection or query-window tuning. Assess prediction errors and mode
ranking before full-history replay. Treat any fixed-v113-classifier comparison
as an input intervention, not a newly calibrated classifier. Retain all old
failures, normal controls and frozen controls; do not expand seeds or adopt v113.

## Limitations

Oracle centers are diagnostic inputs, not deployment inputs or an identifiability
ceiling. Best competitors can change between arms; fixed true-minus-normal
decompositions are also archived. Standardized intercepts are not priors and
feature magnitudes do not certify extrapolation. All inspected windows and
previously used source holdouts are development evidence. No actor, return or
full-history belief improvement is established. v98 remains retained.
