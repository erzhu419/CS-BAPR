# v104 reference behavior exploration

v103 repaired covariance regressions but left all six prior coverage failures.
Retain v98; change reference collection behavior, not the evaluated policy.

## Matched Collection

- Independent reference seeds: 135230/135231; smoke: 135229. Check separation
  against inherited reference and development populations before collection.
- Each seed: standard/exploratory x high_grav/compound; 6000 steps/path,
  normal prefix through step 3799, unchanged OOD profiles from step 3800.
- Same source seed1 checkpoint (99745 updates), shared replay/context and
  fixed-minus2 updating continuation. Exploratory paths add normalized Gaussian
  action noise with fixed standard deviation 0.15 only after the switch and clip
  to [-1, 1]. The executed normalized action enters replay and recent context.
- Both behaviors: 24000 real transitions/updates, 120000 shadow calls,
  144000 total environment calls. Keep every second suffix input: 4400 inputs
  and 22000 paired five-mode labels per behavior. All four prefixes per seed
  must match exactly. All RNG state is checkpointed for exact continuation.

## Next Analysis

Add each new 4400-input collection to the same incumbent mean bank; freeze
the complete v98 covariance geometry, including its old moment center, error
folds, scales and temporal rules. The primary comparison is new standard vs
new exploratory collection, each using the same 8800-source-index extension union
(incumbent 4400 + new 4400), not an equal-budget claim against v98 alone.
These are source-index rows, not numerically deduplicated state/action pairs.

First inspect mean prediction RMS, support distance and new-reference neighbor
share on the six old failure windows and normal/frozen/historical controls.
Only a useful mean/support improvement justifies calibration and full belief
replay. No actor integration or coefficient search in this round.

## Limitations

This is development data, simulator-known-mode counterfactual supervision,
and two collection seeds from one trained source policy. Collection completion
alone establishes no recognition, return or generalization improvement.

Tasks use resident Linux inputs, no node/GPU preference, and synchronize only
small scalar JSON. Traces, reference payloads and checkpoints stay on server.
