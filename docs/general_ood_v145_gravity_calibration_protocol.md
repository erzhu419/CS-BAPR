# v145 gravity-only learning calibration

Four fresh standard SB3 SAC runs: seeds3/4 crossed with gravity multipliers
1.5 and2.0. All remaining physical multipliers are1; observations, motor limits,
native rewards, healthy termination and timeouts are unchanged. Use the frozen
v144 core/vendor and its standard SAC settings. No labels, context or symbolic
constraints are policy inputs. Keep v144 historical runtimes/artifacts intact.

600k interactions /599745 updates per task; 2.4M interactions /2398980 updates
overall. Native SB3 collection, 255 uniform warmup interactions, one update per
interaction thereafter. Training reset seed*10000+episode index. Report each5k,
save full resumable state and perform matched periodic checks each25k. The
v144 helper gains only real environment/evaluator hooks; its defaults are tested.

Retain both predeclared300k/600k snapshots. Each is evaluated frozen on CPU
in its own gravity AND the unchanged v138 gravity-only peak2.6591479484724942.
The latter is not used for training, checkpoint selection or optimization.
Same resets154000--154004, deterministic plus Gaussian noise137000--137002,
native learned std and1000-step cap: 320 episodes overall. No replay loads or
evaluation updates. Record own and held-out peak separately; do not equate
success in milder training physics with success on the original extreme target.

Two CUDA smokes, one per multiplier:512 interactions /257 updates each, two
256/512-step snapshots, CPU checks on both physics with one reset/two modes.
21 local tests cover old SAC behavior, source-budget alignment, new physics,
unchanged held-out peak, RNG and full-state resume. Only node007, jtl110gpu,
jtl110gpu2 eligible; no preferences/pins/Windows. Small logs/JSON locally,
checkpoints and replay server-side only.

## Interpretation
Learning calibration and frozen transfer, not hidden-regime adaptation or a
CS-BAPR mechanism test. Two coefficients bracket the observed severe-gravity
failure; they are not a hyperparameter ranking. Retain all outcomes.
