# Manuscript audit: 2026-09-30

Scope: the active paper/paper.tex, paper/paper.bbl, paper/appendix.tex, Figure 1 source, and the implementation paths used to interpret the method.

## References
All 30 original entries resolve to real primary records. The revision retains 26 cited references: 21 explicit arXiv versions and five publisher/book records for which no arXiv counterpart was located. Four unused entries were removed. Authors, titles, years, version identifiers, and citation locations are recorded individually in reference_audit.csv.
The selected-version years explain changes such as SINDy (2015), IRMv3 (2020), KANv5 (2025), and SINDy-RLv2 (2025). The v2 SINDy-RL author list includes Lagemann.
NAU means Neural Addition Unit; the title Neural Arithmetic Units names the broader family. The implemented quadratic branch is NMU-inspired, not the published multiplicative NMU.
Citation assessments were corrected for terrain generalization, ambiguity-set robustness, equation learning, physics-informed ReLU failures, and causal invariance. Bus-control and Q-ensemble references now support explicit, relevant statements.

## Mathematical and structural repairs
Replaced the incorrect composition rule by L_H*K_G^2 + K_H*L_G and included tanh curvature. The extrapolation constant is L_err = L_pi + M; NAU layer linearity does not certify the full actor.
Separated dynamics increments, action-valued references, normalized actions, and physical hold duration. Disambiguated entropy/mixing coefficients, mode/path indices, demand/amplitude/wind factors, and boundary/symbolic error budgets. Explained notation and first-use acronyms.
Removed the unsupported replay-to-pointwise PAC claim, unconditional IRM tightening, universal three-environment sufficiency, and the claim that every non-frozen penalty breaks contraction.
Reconnected every appendix proof to the main result; removed hand-written theorem numbers and unavailable Lean-project claims. The core conditional calculus bounds and fixed-operator contraction remain valid under the stated assumptions.
Corrected BC and regularization descriptions, transit overrides, initial sigmoid mixing, and gradient-clipping order against the entrypoints.
Fixed LQR and Hopper best-value highlighting and mismatched improvement percentages. All 166 aggregate values, units, and seed counts are unchanged.
Rebuilt Figure 1 from Table 2 medians, removing invented curvature priors and theoretical return overlays. Kept experimental floats within their sections.
Updated the two shell consumers and package instructions for the figure's table-based input. New experiment summaries remain separate from the historical manuscript plot; both shell files pass syntax checks. No deployment package or experiment was launched.

## Known issues and next work
The historical LeakyReLU arithmetic actor remains uncertified. The revised smooth-nau actor has an explicit whole deterministic-policy derivative bound; it has not been evaluated in the historical tables. No theorem links this output-error bound to episode return or simultaneous actor-critic convergence.
The revised SINDy fit and Torch wrapper consume controls. LQR constructs an action-valued reference from identified dynamics and normalized-action costs; JC now uses full squared Frobenius action-Jacobian error. Historical LQR cells remain exploratory evidence for the earlier implementation.
The prediction-error-variance selector does not establish derivative accuracy or causal identification.
The historical legacy configuration and original transit/LQR/Hopper seed evidence are not fully reconstructible from the retained aggregate CSV. The manuscript package does not contain an executable Lean proof project.
Next: recover the old server's small logs or rerun the frozen historical protocol; evaluate the repaired method in a distinct feedback_v2 study. See docs/symbolic_interface_repair.md for the bounded recovery attempt and run specification.

## Verification
LaTeX compilation and cross-reference resolution passed. All retained bibliography entries are cited; all 18 numbered equations have references; eight tables, one figure, and one algorithm have references. No duplicate labels or unresolved cross-references remain. PDF pages and plot labels were visually inspected; there are no overfull boxes. A nonfatal underfull line remains in the book reference. The method repair passes 12 focused tests, the existing 100-update smoke test, and a short OOD-evaluation CLI run.
