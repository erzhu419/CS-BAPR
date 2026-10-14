# v139 frozen replay diagnostics

Six final300k full-state checkpoints: v138 friction seed3 and gravity seed4
(divergent), friction seed4 and gravity seed3 (same-profile numerical controls),
v137 ID seeds3/4. No new interactions, optimizer steps or environment rollouts.
Frozen core inherited from v138; no algorithm edits or checkpoint selection.

CPU probes draw128 transitions without replacement from each collection-step
interval:1--25k,25k--50k,50k--100k,100k--200k,200k--300k. Private NumPy seed139000
and torch seed139001 fix indices and actor noise; smoke uses8 per interval.
Read saved states/contexts, compare recorded-action and policy-action ensemble Q,
target/current critic gap, and decompose the MSE target into normalized reward,
discounted min target Q and entropy. Use the saved final reward EMA std without
updating it. Alpha-zero and stable-density targets are diagnostic counterfactuals.

Native density exactly follows the actor's epsilon-regularized tanh formula.
Compare against PyTorch TanhTransform log Jacobian, separately checking Normal
evaluation against the reparameterized standard-noise density. Measure saturation,
log-std limits and gradients at detached mean/log-std heads; no parameter updates.

Full replay/checkpoints remain on Linux servers; only small JSON results/logs
return locally. CPU2/RAM8192MB, all three Linux nodes eligible, no preferred node,
GPU or hard pin. Native divergent/control smoke acceptance precedes formal runs.

## Limitations
Only final weights exist: collection-age strata are not historical model states.
These probes locate late-model pathologies, not the historical causal root or a
physical controllability limit. A counterfactual target is not a validated fix.
