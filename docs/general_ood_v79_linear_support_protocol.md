# v79 Linear Support Attribution

Reuse all 65 saved v78 paths, each first50/last64 window. No simulation,
reference refit, recalibration, temporal change, actor feedback or new seeds.
All traces and arrays stay on the server; retrieve scalar JSON/logs only.

Measure nearest/64th-neighbor distances, standardized state/action offsets,
query-to-centroid distance, ridge query leverage and signed prediction weights.
The query leverage is 1/64 + d' (X'X + 0.064 I)^(-1) d, where X is the centered
local reference design and d is its centroid minus the query. The intercept
remains unpenalized. Record local fit and leave-one-out residual RMS per mode.

Under each unchanged covariance, decompose the true-mode energy improvement
as 2 e' P c - c' P c: constant-mean error e, precision P, linear correction c.
Record coordinate contributions and matched competitor gains/margins.
Verify window prediction RMS and density margins against v78 before use;
native smoke also verifies predictions against the original v78 implementation.

Next decision: distinguish unsupported corrections from covariance-weighted
ranking failures before choosing a reference-only correction. Correlation is
diagnostic, not a causal claim or an acceptance rule. No fresh expansion yet.
