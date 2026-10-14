# v122: target L1 scale and single-factor source pilot

First decompose the three frozen v121 final resumable checkpoints on their native node.
Use 512 evenly spaced transitions from the full replay and from the last 10000
interactions, with three policy-noise seeds 12200--12202. Reconstruct the actual
critic loss and report reward, discounted entropy, min-next-Q and target L1 uplift.
This probe has zero optimizer updates and environment steps; it establishes scale,
not training causality. Proceed with the pilot if the uplift dominates immediate reward.

Pilot: three fresh normal-only training seeds 0/1/2, each 100000 interactions.
Change only `weight_reg: 0.01 -> 0.0`, removing the additive target-critic L1 uplift.
Retain actor beta=-2, critic disagreement penalty, behavior cloning, context,
reward normalization, auto-alpha, exploration schedule, original reward and physics.
Reuse v121's exact training/evaluation/checkpoint driver and matched completed seed controls.
Do not call this vanilla SAC or adoption of a new final method.

Final saved model only; five evaluation episodes on seeds 150000--150004.
Judge forward velocity jointly with survival; retain every training seed and cost.
Record final target-scale decomposition as well as the unchanged periodic evaluations.
Keep the original software stack and let scheduler choose GPUs without preferences;
no Windows. Synchronize scalar JSON only, keeping all replay and checkpoints native.

If removing the uplift reduces Q scale but does not recover locomotion, do not
declare success: test exploration or reward/entropy balance next. Rebuild source-matched
readouts only after useful source locomotion is established; v119/v121 negative results remain.
