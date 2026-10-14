# v54: normal-calibrated signed residual windows

Freeze v53 preflight's broad quadratic dynamics model and scalar reference. Use
all11 standardized residual coordinates with their signs intact. At each physical
transition, average the last16 residual vectors (expanding history initially;
never clear at an episode reset). Fit the vector mean and full covariance on
normal-only moment-fit trajectories. Regularize covariance by adding
1e-3*mean(diagonal covariance)*I, fixed before test results. The squared centered
Mahalanobis score is divided by11; no coordinate is selected using OOD results.

On separate normal calibration traces, compute causal window scores, their global
q95, then psi=log(1+score/q95) and the unchanged normal-age q95 reference. Retain
the existing16-step mean of instantaneous departures for the final coefficient.
The residual window and departure history both span episode boundaries. Only
independent trajectories clear history. This changes the readout, not dynamics,
observations, physics, agent losses, replay sampling or scalar baseline thresholds.

Every trajectory starts from v40 fixed-minus2 seed1 and v52's original fit-only
normal replay/context, using stochastic actions, beta=-2 and TimeLimit1000. Moment
fit/calibration each include frozen and normal-updating behaviors. New reference
traces do not augment test replay. Preflight: fit98400, cal98410, test98420/1;
formal: fit98500/1, cal98510/1, test98520--323. All traces1200 interactions; each
test seed has normal/heavy/slippery/high_grav/compound branches with a step800
physical change and exact paired prefixes/switch inputs. Readouts never affect
the policy, and references remain frozen throughout testing.

Record v53 scalar and signed-window full/prefix/first-post50/post/last64 statistics
and paired contrasts. Retain v53's pass condition: every learning-normal full and
last64 coefficient<=.10, and every changed profile/seed's first50 and last64 paired
contrast>0. No threshold search using returns or test traces. Preflight costs
16800 new interactions/14400 updates; formal33600/28800. Save full trajectory and
experiment progress every500 steps and between cells, reusing v53's native-state
resume. Only scalar JSON/logs return locally; large data remain server-side.

Existing inputs reside on node007; scheduler chooses its idle GPU, no GPU
preference or Windows. Formal expansion requires preflight readout validity.
Limitations: one source seed, two/four conditional test seeds; no adaptive RL or
50x OOD return confirmation. The doubled history can delay responses relative
to the instantaneous scalar baseline.
