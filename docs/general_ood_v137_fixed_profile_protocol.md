# v137 fixed-profile learnability

Fresh policies for ID/easy_surface/mega_event, training seeds3/4,300k native
interactions each: six runs /1.8M interactions /1798470 optimizer updates.
Use the frozen v136 core and v132 MSE agent/config, corrected terminated-only
bootstrap masks, online transition context, effective beta-2 and weight_reg0.
No NAU, symbolic reference fitting, SINDy or symbolic/Jacobian constraints.
All six runs use the same uninterrupted training driver and budget, including
fresh ID controls; historical staged ID sources remain separate.

Keep structured intensity50 and exact existing mass/gravity/friction values.
Activate one static physical profile, then use the native Hopper TimeLimit
for training/checkpoint restoration. Confirm the parameters persist across
resets and resume. Native observations, actions, rewards and health termination
are unchanged; profile labels are not policy inputs. Periodic deterministic
evaluations use the matching profile, preserving context and CPU/CUDA RNG.
They are diagnostics and never select a checkpoint or update optimizers.

Evaluate restored final300k snapshots on CPU in their matching static profile,
resets154000--154004, deterministic plus common native-scale action noises
137000--137002:20 episodes per policy /120 total. Restore source normalizers,
reset EMA and update real transitions per episode. No replay in final inference.
This is known-regime learnability, not unknown-regime adaptation or a baseline
comparison. Success warrants regime-randomized training; failure at300k is not
a physical-impossibility proof.

Before release, verify native/profile parity, unchanged ID training, matching
periodic evaluation, timeout masks and exact full-state resume. Native CUDA
smokes cover all three profiles with512 steps /257 updates each. Eligible
node007/jtl110gpu/jtl110gpu2, no Windows, preferences or pins; CPU2/RAM4096MB/
VRAM1024MB. Only small source.json/logs pull locally; final/resumable snapshots
move server-to-server. Historical results, actors and paper tables unchanged.
