# v109: reference-model density uncertainty

v108 hard selection reduces normal confusion but retains a recognition/risk
tradeoff. Test two structures with fixed uniform weights: mix old/new normal
density while using new OOD densities; or mix old/new densities for every mode.
For each mixed mode use `logaddexp(old_ll, new_ll) - log(2)`, the log of a
normalized probability mixture. Do not average log densities or posteriors.

The component is marginalized independently at each transition. This is not
persistent latent-branch inference, a fitted weight, or trajectory-label routing.
Keep old, new and old-normal/new-OOD as controls. Each of the five assemblies
gets its own unchanged calibration procedure using the original 12,000 normal
transitions (seed135060 frozen/updating). Reproduce nine control calibrations.

Smoke covers four seed135192 normal/compound frozen/updating trajectories and
36 control path-condition comparisons. Formal replay covers all 95 paths /
570,000 cached transitions, with 855 control comparisons, three covariance
conditions and complete belief histories. Report the same populations, old
failure windows, historical slippery tail and primary normal episode diagnostics.
No new density/geometry computation, environment interactions, actor integration
or weight search. Only scalar JSON is synchronized; phase objects stay native.

## Limitations

Development-only evidence and one source policy. The two models share reference
data and are not independent members; reference budgets differ. Calibration
reuses the original population. No return claim; risk means are not error rates.
