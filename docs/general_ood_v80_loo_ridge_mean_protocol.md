# v80 Reference-Only Ridge Selection

Change only the v78 updating-bank mean rule. Keep the same 64 neighbors,
23,000 reference labels per mode, forward/scaling, both covariance banks,
recursive hazard and reset/age rules. Replay the same 65 development paths.

At each query state/action, fit all five residual vectors with one shared
ridge strength, selected by minimum reference leave-one-out MSE across all
64 neighbors and 55 normalized outputs. Candidate mean-loss coefficients
are 0.001, 0.01, 0.1 and 1 (ridge alpha 0.064, 0.64, 6.4 and 64). The intercept
is unpenalized; exact ties select the stronger ridge. Leave-one-out keeps the
query-selected neighborhood and candidate alpha fixed while removing a row.
Neither query next-state nor true mode is available to the selector.

Compare selected means with frozen/updating covariance, retaining all five
v78 controls by reusing their saved statistics. Recalibrate both new readouts
on the same 12,000 normal-null transitions. Record per-window strength
fractions, reference losses, weight norms, prediction errors and mode margins.

Before full replay: focused tests and native smoke on v76 compound seed135180
and v61 heavy seed132020, checking 228 transitions against sklearn Ridge,
batch/standalone density equivalence and unchanged constant controls.
Resume only completed path prefixes; phase/progress files stay server-side.

Decision: reduce early regressions without sacrificing normal, frozen or
historical paths. Retain identification failures. No fresh seeds, simulation,
actor feedback or paper-table changes; scalar JSON/logs only, no Windows jobs.
