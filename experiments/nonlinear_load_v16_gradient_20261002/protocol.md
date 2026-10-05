# Nonlinear load v16: read-only gradient diagnosis

v15 rejected wider training resets: both seeds worsened initial regulation. Stop that intervention and diagnose the four existing last candidate checkpoints, balanced/narrow seed9/10. No training, fitting, optimizer updates, new seeds or manuscript changes. Historical performance gates and HOLD remain unchanged.

## Frozen Measurement

Use the v15 runtime and existing paired evaluation paths 60000--60099, radii 0.2/0.5/1.0, severities 1/50. Reproduce actor returns before using their state banks. At each radius/severity, compare both actors on identical initial states and a shared mixture of both actors' last-150-step trajectories. Separately compare both actors on each source policy's replay samples, keeping source distributions explicit.

On the checkpoint servers, summarize all 100,000 replay transitions, first 50/last 150 steps, and the final 100 training episodes. Record physical/load norm quantiles, physical norm bins, mean state cost and reset support per training radius. Export only actor/critic parameters, terminal alpha/belief coefficient, fitted gain, aggregate support and four 256-state replay batches. Do not download full checkpoints, optimizer state or full replay.

Sample replay banks with replacement using NumPy seeds 91000--91003 in all/early/late/recent order. Initial and shared late-trajectory banks each use seed91000 and size256. Evaluate eight common Torch noise draws, 92000--92007, using the native actor.evaluate. A CUDA preflight uses seed9, radius1/load50, path60002, all-replay banks and the first two noise draws; it is not performance evidence.

Decompose the frozen actor objective into negative mean Q, risk penalty, entropy and 0.01-weighted Jacobian error. Split the latter into physical-input and load-input columns with exactly the native loss normalization. Record losses, pre-clipping parameter gradient norms, SAC/Jacobian cosines, total-gradient clip scale and derivatives of initial physical-Jacobian error under each replay gradient's descent direction. Positive derivatives mean worsening. Average vectors across the eight draws before computing mean-gradient relations; retain individual draws.

Validate native loss/gradient equivalence and non-mutation in focused tests. Preflight must reproduce saved actor returns and pass native Jacobian equivalence, finite gradients and unchanged actor/critic parameters. A failure stops formal dispatch for repair. Two formal tasks, one per existing training seed, use idle compatible Linux GPUs without machine/GPU preferences or Windows nodes.

## Interpretation And Limitations

This is a local diagnostic of terminal frozen actor/critic/alpha/belief, not reconstruction of historical updates, causal attribution or a predicted Adam step. Adam moments and weight decay are excluded; clipping scale is reported but no clipping/update is executed. Replay norm marginals describe sample weighting, not full multidimensional support. State-matched gradients on OOD states may use inaccurate critics. An optimization-conflict interpretation needs the same directional pattern on both training seeds and explicit replay-source/phase comparisons. No new performance threshold or parameter sweep is introduced.

## Execution

Three focused tests passed locally and in the frozen runtime. CUDA preflight t125003 passed; formal seed9/10 tasks t125006/t125007 exited 0 after 20.7/21.8 seconds. All native-loss, reproduction and parameter non-mutation checks passed. The proposed persistent terminal combined-gradient mechanism was not supported; performance HOLD is unchanged. See docs/nonlinear_load_v16_gradient_results.md for the scoped next experiment.
