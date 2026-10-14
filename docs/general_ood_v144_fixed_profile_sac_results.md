# v144 standard-SAC learnability results

All six formal tasks t148407--t148412 completed exit0. Training: 3.6M
interactions /3598470 updates. Both predeclared snapshots and all matched
physics, reset/noise cells and update budgets verified. 240 CPU episodes /
111776 steps; 1548533 JSON bytes pulled, no checkpoints/replay. Runs took
138.1--195.2 minutes including milestone evaluation, excluding publication.

| Profile | Seed3 sampled length 300k ->600k /15 | Seed4 sampled length 300k ->600k /15 |
|---|---:|---:|
| ID | 974.07 ->1000 | 419.87 ->1000 |
| Friction-only | 300.20 ->351.40 | 619.73 ->396.47 |
| Gravity-only | 82.73 ->84.07 | 83.87 ->85.53 |

At600k, both ID policies finish all20 episodes without falling. Low friction
improves over the v142 learner's125/153 sampled steps, but all40 final episodes
still fall; seed4 deteriorates from300k to600k. Retain this negative result,
not a selected intermediate checkpoint. The fixed gravity multiplier2.659
produces80 falls across two seeds/two budgets, all height terminations; late
periodic checks remain around83--87 steps. No blanket physical-impossibility
conclusion follows, but further budget extension of this setup is not justified.

Next v145: change only training gravity to1.5/2.0, seeds3/4, same standard SAC
and600k budget. Evaluate both300k/600k on own physics AND the unchanged2.659
held-out gravity peak. Keep friction, motor limits, rewards and health thresholds.

## Limitations
Two independent training seeds. Resets/noise repeats are not additional seeds.
SB3 differs from the earlier learner in several implementation details; these
controls establish bounded learnability evidence, not a mechanism-only ablation
or general algorithm ranking. Low-friction stability remains incomplete.
