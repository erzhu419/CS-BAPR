# v120 mechanism diagnosis

No training, refitting or threshold/gain tuning. Reuse all60 native v119 traces
and the one source checkpoint. Three seed shards135270/271/272; CPU-only on
node007 where all inputs reside, no Windows or placement preference.

1. Decompose every real trace's rewards into healthy, forward and control
   components using actual Hopper-v4 settings. Under terminate_when_unhealthy,
   healthy reward is1 even on terminal steps. Reconstruct interval-average
   forward velocity from reward, not clipped instantaneous observation velocity.
   Match the existing v119 reward/risk summaries.
2. Evaluate the unchanged source (update99745) from the same shared context
   snapshot, all five modes, same v119 deterministic evaluation seeds and three
   max1000-step episodes each. Compare reconstructed velocity against native
   info[x_velocity] on every step. No replay insertion or optimizer update.
3. Use one common fixed-minus2 compound replay per seed. Select256 evenly spaced
   samples from prechange3500:3800 and tail5700:6000, including their saved online
   contexts. Compare source and four final compound-endpoint actors/critics on
   exactly those inputs, with three fixed shared policy-noise draws12000/1/2.
   Measure Q std, alpha, reward scale, actor-loss gradient components and the raw
   gradient change between beta=-2 and beta=-3. Noise repeats are not extra seeds.

The gradient calculation is before clipping, Adam and weight decay. It does not
run an optimizer or predict a return improvement. Other v119 final-mode actors
were overwritten by latest.pt, so no claim reconstructing those models or past
training gradients. Source evaluation cannot establish cross-source robustness.

Native smoke uses seed135270, eight existing normal/compound traces, one max64
episode per source mode,16 gradient samples and one noise draw. Small progress
checkpoints enable resume; only scalar JSON is synced. Formal summary command:
`python3 scripts/summarize_general_ood_v120.py --result-dir results/general_ood_v120_mechanism`.

Decision after diagnosis: if survival/near-zero velocity already characterizes
the source, repair the locomotion training protocol before expanding belief
experiments; if uncertainty-gradient leverage is weak, scalar risk alone is an
insufficient intervention. No automatic algorithm adoption or larger seed sweep.
