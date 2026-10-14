# v59 normal-only attribution

Two tasks, each comparing frozen-parameter and updating branches over6000
normal Hopper-v4 interactions. Reuse v58 continuation116000/116001 deliberately
for counterfactual attribution, not as new independent seeds. Total24000
interactions,12000 optimizer updates. Same source/replay/context, stochastic
actions, normal physical parameters and reset seeds as v58; beta=-2 throughout.
The first transition must equal the completed v58 fixed-minus2 result.

Read both v57 formal pooled and phase references on each identical branch trace.
Neither sensor controls beta, and no model/reference is refitted. Report250-step
bins, first800/full6000/last1000/last64 summaries, episodes and actor parameter
movement. No return-based parameter selection or new RL performance claims.

Save full native/replay/RNG progress every500 interactions, with raw traces and
checkpoints kept server-side. Pull only scalar JSON/logs. Normal-only learning
drift would motivate calibration across the learning lifetime; a clean normal
counterfactual would instead motivate post-OOD persistence diagnostics.
