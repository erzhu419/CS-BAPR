# Nonlinear load v9 pilot results

The frozen pilot completed on local Linux through scheduler `profile-local`: exit 0, 49 seconds, peak RAM 455 MB, VRAM 0. All 4,200 episodes completed on the specified 100 paired paths at six severities. Identification/validation used 20,000/2,000 interactions; RL runs and optimizer updates were zero. Nine focused tests passed. Full small-artifact evidence: `results/nonlinear_load_v9/pilot.json`, `completion.json`, and `experiments/nonlinear_load_v9_20261002/formal.log`.

Mean total cost, lower is better. The two fitted columns are independent identification models, not trained RL agents.

| Severity | Zero | Known local-linear | Known inverse feedback | Fitted 91000 | Fitted 92000 |
|---|---:|---:|---:|---:|---:|
| 1 | 90.327265 | 0.255739 | 0.275984 | 0.260954 | 0.261000 |
| 2 | 92.207576 | 0.257130 | 0.277383 | 0.263103 | 0.263172 |
| 5 | 103.226473 | 0.269106 | 0.289153 | 0.283732 | 0.284335 |
| 10 | 187.767770 | 0.322244 | 0.341993 | 0.373285 | 0.379748 |
| 20 | 709.780139 | 0.597803 | 0.611196 | 0.853170 | 0.875210 |
| 50 | 5672.390222 | 2.859103 | 2.726351 | 4.926836 | 5.284658 |

Both readiness gates pass. Inverse feedback beats zero at every severity; its 50x cost is 0.0481% of zero, maximum sampled physical norm is below 0.279, and mean component saturation is zero throughout. Both unchanged fitted-reference builders succeed. The environment and existing reference interface are ready for a learner test.

The fitted models imply near-origin force gains 3.994--4.021, versus the true gain 2.5. For uniform random commands, the best linear fit to `2.5*u + 2.5*u^3` has slope `2.5 + 2.5*E[u^4]/E[u^2] = 4`; this explains the observed slope without fitting another model. At 50x, inverse feedback costs 46.6% less than the mean of the two fitted controls and wins 99/100 and 100/100 paired paths. However, known local-linear feedback already closes most of this gap: inverse feedback improves on it by only 4.64% and is worse at severities 1--20. Keep all these controls in the learner comparison.

## Next Step

Freeze the environment without altering actuator coefficients, reward, loads or gates. Plan eight training runs: seeds 5/6 for the v8 centered/no-NAU symbolic candidate, its no-SINDy counterpart, BAPR and RE-SAC. Keep the 100,000-interaction budget, final-checkpoint evaluation and v8 weights; evaluate all six severities on paths 50000--50099. Each symbolic run must evaluate its own saved fitted feedback on the same paths; known local-linear and inverse feedback remain distinct privileged controls. Report added RL value against the learner's own fitted feedback, separately from learner rankings and known-model gaps. No RL tasks were submitted by this pilot.

Scheduler placement must exclude Windows following the user's instruction, while leaving all compatible Linux machines eligible without node/GPU preference. Remote attempt `t119827` failed during Windows bootstrap before evaluation; its automatic retry `t119861` was cancelled. The local formal run used the existing pinned runtime, not the incomplete Windows environment. No trajectories or checkpoints were pulled, and v7/v8 evidence and manuscript tables were left unchanged.

## Limitations

Readiness is a sampled finite-horizon controllability result, not a global stability certificate. Known-model feedback is feasible, not optimal. This remains a synthetic nonlinear-actuator task with observed persistent load, not evidence of standard-benchmark generalization. Validation increment R-squared 0.976--0.978 does not eliminate the demonstrated local derivative bias. No learner has yet shown it can improve the fitted controller or preserve the benefit of the unchanged symbolic penalties here.
