# v135 frozen OOD/context test

Freeze both MSE final300k policies. Compare online context against an all-zero
context input using the same weights, native state interface and rewards.
Zero context is an inference ablation, not a representation-training ablation.
Restore each source normalizer and reset its EMA before every episode. Online
updates from real transitions; zero context neither observes nor supplies
context signals. Policy parameters, optimizers and replay remain unchanged.

Conditions: ID; four static structured profiles (easy_surface, heavy_traffic,
mega_event, slippery_rush); Markov profiles switching every50 steps with
probability1. The first Markov change affects step51, after50 ID steps.
Intensity50 uses the existing profile generator; record the actual body-mass,
gravity and friction multipliers, not a claim of50x each parameter. Sensor,
actuator and reward noise/shaping are disabled. Policy action sampling stays
at native scale1 using the v126 common CPU noise sequence. Labels enter logs,
never policy inputs. Log non-ID exposure, switches and physical failure causes.

Five resets152000--152004, deterministic plus three sampled action-noise
seeds135000--135002,1000-step limit:120 episodes per seed/context pair,
four tasks /480 episodes, at most480000 evaluation steps. No new training or
optimizer updates. Include the ID ablation to distinguish a general context
dependency from an OOD-specific effect. This is not a RE-SAC/BAPR comparison.

Before release, test context zeroing/restoration, physical switches/native
rewards, common noise and episode-level resume. Two native CPU smokes evaluate
ID/static-mega/Markov in both action modes,128-step cutoff. Formal tasks retain
the1000-step protocol. Only final.pt stages between servers, never replay.
CPU2 /RAM2048MB /VRAM0, eligible node007/jtl110gpu/jtl110gpu2 without preferences
or Windows nodes. Only small ood_context.json files sync locally. Older
sources, environment implementations, results and paper tables stay unchanged.
