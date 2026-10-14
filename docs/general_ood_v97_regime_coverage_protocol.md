# v97: equal-budget regime-diverse reference collection

Stage 1 collects only references. Seeds 135200/135201 each produce four updating paths from the same source-policy checkpoint (training seed1, 99745 updates; shared replay/context bootstrap). All paths remain normal until step3800 and continue to step6000 using fixed-minus2. Old recipe: normal/slippery. Diverse recipe: high-grav/compound. The modes use unchanged known mild profiles. Both recipes therefore receive 24000 real transitions, 120000 shadow calls, 144000 total simulator calls and 24000 optimizer updates. All four paths at each seed must have the same pre-switch prefix.

Each recipe takes every second post-switch transition: 1100 inputs per path, 4400 total inputs and 22000 paired counterfactual labels. Every input carries all five known-mode next-state targets; only real transitions enter RL replay. Reference seeds are disjoint from prior references and evaluated paths, including v95. No v95 trajectory or target enters the new reference payload.

Native smoke: seed135199, four 64-step paths switching at32; 256 real transitions/updates and 32 reference inputs per recipe. Check exact true-mode replay, matched prefixes, equal budgets and paired labels before formal dispatch. Checkpoints, traces and payloads stay on node007; only scalar JSON is synced. No Windows placement or soft node/GPU preference.

Stage 2, after both complete payloads are verified: extend the same v94 raw/mean banks with equal row counts; recompute whole-seed-excluded errors, retain the raw-mixture weight/disagreement coefficient and model/covariance forms, and recalibrate both recipes on the same 12000 normal-null transitions. Replay all v95 paths and existing development controls, including full normal windows. Preserve the unchanged v94 control. Fresh confirmation and actor integration remain outside this stage.

## Limitations

Known-mode simulator labels and one source policy; collection alone is not recognition or return evidence. Post-switch collection regimes change the state/action distribution. v95 now supplies development evaluation, not fresh confirmation of the revised references. Stage 2 is planned, not yet dispatched.
