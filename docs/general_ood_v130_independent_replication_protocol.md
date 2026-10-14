# v130 independent mean-retention replication

Add training seeds 3/4 without changing the v122/v123/v128 learning protocols.
For each seed, run v122's 100k L1-off source, continue unchanged to v123's final
300k source, then fork that complete snapshot into two 50k paths: unchanged
control and mean-only source regularization with coefficient one. Keep beta=-2,
weight_reg=0, original Hopper-v4 reward/physics and context updates. No random
action warmup, parameter sweep, filtering of weak sources, source readout refit
or selection of a better periodic model. The only training-entry edits expand
the accepted command-line seeds to 3/4; training functions and stage specs stay
unchanged. Source/result revision fields retain their original protocol names.

Eight formal GPU tasks: two 100k prefixes, two 200k extensions, four 50k continuations.
New training interactions: 800000; expected updates: 799490 with batch size256.
Every continuation restores the source's actor/critics/optimizers/replay/context/
RNG/environment/unfinished episode. Each arm uses its own seed's final 300k
actor as the frozen reference. Existing seeds0/1/2 use the same v123/v128 final
models as v129, without retraining or deleting negative results.

Evaluate all five training seeds on CPU with new reset seeds151000--151004 and
new noise seeds14000--14019. Use the native 1000-step limit and v126's independent
CPU float32 (1,3) Gaussian draw per step, noise scale one. Each model starts
with its native context normalizer, resets EMA per episode, updates on actual
transitions and restores context/RNG afterward. Models are final 300k reference,
350k control and 350k retention: five deterministic plus100 sampled episodes
each. Five CPU tasks /15 models /1575 final episodes /zero evaluation updates.
The new deterministic resets are evaluation cells, not reproductions of v129's
old baseline cells. Report per-training-seed matched 350k changes and noise-wise
pairs; the 300k reference provides context, not a matched-budget competitor.

Scheduler dependencies use only synced small completion JSON: prefix -> source
-> both continuations -> each new seed's evaluation. Existing-seed evaluation
has no new training dependency. Dedicated native checkpoint paths support exact
resume; frozen evaluation resumes at episode boundaries. No checkpoint/replay
download, only scalar JSON. Freeze v129 code plus the three seed-entry edits
and the v130 evaluator/merger. Eligible nodes are Linux node007, jtl110gpu and
jtl110gpu2, without machine/GPU preferences or Windows nodes. Both added nodes
receive the packed native Python3.10.14/NumPy2.1.3/Gymnasium1.2.0/MuJoCo3.2.7
base, with Torch2.5.1+cu121 and CUDA dependencies installed from official package
sources at their native pinned versions. Unused JAX packages are not deployed.
Their environments and v130 runtime live
on /data or /mnt/data, not jtl110gpu's nearly full system disk.

The multi-node entry stages complete 300k source snapshots directly between
servers before a continuation, or final actors alone before frozen evaluation.
Completed continuation paths publish only final actors and scalar JSON to the
central HPC store before scheduler completion; their latest resumable snapshots
and replay remain on the compute node. JSON path fields are mapped between
canonical local-node and central-store paths without changing training/eval
values. The original training functions, stage specs and coefficient remain
unchanged. Source extensions keep their existing progress on node007;
waiting continuations/evaluations use the scheduler's three-node resource pool.
Each added node must pass CUDA, full-snapshot continuation and frozen-evaluation
smokes before the dependent formal tasks are released. Record compute nodes and
GPU models with results; this round is no longer a same-GPU execution.

Environment configuration is tracked by scheduler tasks on each added node.
Their small environment.json results gate the per-node smokes; a local acceptance
task checks two 512-update CUDA continuations and two six-cell frozen evaluations
against the native seed2 smoke before writing multinode_ready.json. That marker
releases the six existing formal tasks automatically. New-node child processes
prefer their pinned CUDA libraries over scheduler-injected JAX library paths.
The exact frozen runtime entries are explicitly staged on both nodes; these
tasks then skip cached scheduler source staging. No model-selection decision or
formal training budget is changed by setup or smoke interactions.

Native smoke: seed3's original 512-step prefix and seed2's six-cell frozen
evaluation with one new reset/noise. Smoke is an execution check, not model
selection or hyperparameter tuning; its interactions are separate from formal
cost. Final evidence distinguishes the two new replication seeds from the
three seeds used to develop the intervention. This remains normal-Hopper source
robustness, not an OOD or submission-readiness claim.
