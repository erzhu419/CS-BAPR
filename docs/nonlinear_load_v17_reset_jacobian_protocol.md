# Nonlinear load v17: reset-balanced Jacobian sampling

v16 found active Jacobian correction, not a terminal combined gradient that persistently worsens initial alignment. Test sample weighting directly. Run the unchanged candidate on fresh matched seeds11/12 with ordinary replay Jacobian states versus a 50/50 reset/replay Jacobian batch. Four formal runs; no historical pooling or new comparator expansion.

## Frozen Intervention

Both arms use the v15 centered tanh MLP, no NAU, own fitted linear feedback, Jacobian weight 0.01, no target shift, hidden size 64/three critics and all other v15 settings. Keep the SAC replay sampler, its batch size, and surprise/alpha/critic update rules unchanged. In the intervention, the first 128 SAC replay states supply the ordinary Jacobian half; the other 128 are sampled uniformly with replacement from stored reset observations. Reset indices are episode multiples of 200 in the unwrapped replay. Include the current episode's reset once stored. All reset radii are eligible; no severity-based filtering.

The reset sampler uses its own NumPy Generator, seeded100000+training_seed. It consumes no global NumPy, Python, Torch, environment or action-space RNG draws. Save its state and sample counters in every full training checkpoint. No extra interactions, dynamics fitting or state augmentation. The control's Jacobian state hook is identity.

Both arms train at severity1 for500 episodes/100000 interactions/99001 updates, using the absolute-episode 0.2/0.5/1.0 reset cycle (167/167/166 episodes). Each arm uses the unchanged separate radius0.2 SINDy fit/validation:10000/1000 interactions. Reset training RNGs after identification. Keep only last checkpoint for evaluation and the existing20-episode save interval. Frozen Python3.10/3.11 and Torch2.5.1+cu121 numerical runtime.

## Evaluation And Decision

Evaluate all three reset radii at severities1/50 on the existing100 paired paths60000--60099, with actor, own fitted, known local-linear and inverse-actuator controls. Save first50/last150-step state/command costs and initial/trajectory Jacobian alignment. No learning or fitting during evaluation.

Sampling repair requires lower radius1 total and first50-step state cost than the matched replay control on each seed and severity. Small-radius retention requires radius0.2 total cost no greater than the matched control on each seed and severity, with zero noninferiority margin. Own-fitted added value requires lower candidate cost than own fitted feedback on both seeds at both severities, reported separately at each radius. Proceed to fresh learner-comparator replication only if repair, small-radius retention and all-radius own-fitted added value pass. Otherwise retain HOLD without retuning the ratio, weight, seeds or checkpoints. Historical v13/v15 HOLD is unchanged; this two-arm study cannot establish learner-baseline superiority.

## Verification And Operations

Focused tests detect incorrect reset indexing/mixture, global RNG consumption, changes in the control loss, and lost sampler state or learner updates after resume. Failures stop dispatch for repair. Two six-episode CUDA preflights (both arms, seed11) checkpoint/reload after episode5 before the final200 interactions, then evaluate all blocks on two paths. Require201 updates and25728 reset plus25728 ordinary Jacobian samples in the intervention. Preflights are not performance evidence.

Scheduler chooses idle compatible Linux GPUs without machine/GPU preference, pin or allowlist; Windows nodes are excluded. Full checkpoints remain outside result directories on servers. Fetch only small logs/JSONs. No runtime optimization or manuscript edits.

## Limitations

The sampling ratio is a new intervention, not a proven fix. Both arms already cover larger resets in training, so this is not unseen-reset transfer. Two fresh seeds and reused disturbance paths limit inference; OOD load stress remains severity50. Direct controllers are feasible references, not certified optima.

## Dispatch

All 24 focused tests passed locally and in the frozen runtime. CUDA preflights t125028/t125029 exited 0 after native checkpoint/reload at episode 5, with 201 updates; the intervention retained 25,728 reset and 25,728 ordinary samples. Matched exploration histories, fitted gains and direct-control returns passed. Only 224,201 bytes of preflight result JSON were needed; no checkpoints were downloaded.

Formal tasks t125062/t125063 are replay/reset-balanced seed11; t125064/t125065 are replay/reset-balanced seed12. All four exited 0 on scheduler-selected node007 and jtl110gpu, without placement restrictions, in approximately 34--51 minutes. Source, manifest, task specifications and startup observations are in experiments/nonlinear_load_v17_reset_jacobian_20261002. The frozen summary passed formal data validation; repair, retention and all-radius own-fitted gates failed. See [results](nonlinear_load_v17_reset_jacobian_results.md). Retain HOLD and leave manuscript tables unchanged.
