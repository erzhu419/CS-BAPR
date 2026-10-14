# v131 frozen source diagnosis

Use six existing final full-state sources: seeds3/4 at100k, and seeds1/2/3/4
at300k. The original latest.pt contains the final actor/critics, reward EMA and
replay; the lightweight final.pt alone is insufficient. Freeze the v130 runtime
and add only the probe/transport entries. No training, optimizer step, simulator
interaction, reference refit or checkpoint selection.

For each source, use4096 evenly spaced replay indices across its complete replay
and4096 across its last10k transitions. Repeat the actor draws with fixed noise
seeds13100/13101/13102. Each pool is processed in native256-transition batches;
reward normalization is computed per batch from the frozen EMA without advancing
it. Reconstruct the native conservative critic loss at every batch. Alpha and
all model parameters stay frozen. Actor gradients use the frozen critic, not a
critic updated by a hypothetical training step.

Report per-critic absolute TD-error quantiles, mean squared residuals, top1%/10%
transition loss shares, replay-done/non-done groups, five worst sampled
transitions, target/entropy/normalization/input/action/log-probability scales,
and total/component actor/critic gradient norms before clipping. autograd.grad
leaves parameter gradients unset. One native smoke uses the100k seed3 source,
256 samples per pool and the first noise seed; its small result gates six
formal source tasks. The smoke is an execution check, not source selection.

Scheduler eligibility: node007, jtl110gpu and jtl110gpu2, no node/GPU preferences,
no Windows nodes. CPU2 / RAM4096MB / VRAM0 per task. Native environments already
passed v130 acceptance. Inputs move directly between servers using v130's
transport; only scalar source_diagnostics.json returns locally. Frozen source
snapshots remain on the servers. No environment installation or training change.

## Limitations

Replay stores terminated-or-truncated, so done groups cannot distinguish physical
termination from time limits. Fixed replay slices and noise repeats diagnose
endpoint losses, not training causality, generalization or independent seeds.
Seed3's instability starts before100k; these are available predeclared endpoints,
not a reconstructed onset checkpoint. Seeds1/2 provide useful-source contrasts;
their replay distributions differ, so cross-source scales are not paired effects.
