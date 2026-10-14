# v143 passive terminal-window diagnosis

Eight CPU tasks, zero optimizer updates/new training interactions. Six v142
final600k models evaluated in their own static profiles; the stable ID seed4
policy also evaluated frozen in friction-only and gravity-only. Same resets
154000--154004, action-noise seeds137000--137002 and1000-step cap:160 episodes.
Restore the trained context normalizer and reset the short EMA per episode;
update context from observed transitions exactly as in the existing evaluator.

Record normalized command effort/saturation and the last17 sampled snapshots,
stride4 plus termination: pre/post qpos/qvel, actual commands and generalized
actuator joint torques. State labels and control ranges/gear come from MuJoCo.
Terminal reasons are already known from v142 JSON; this round adds the missing
control trajectory and cross-physics positive controls, not another reason audit.

Preserve native rewards, healthy termination and all profile physics. Instrument
the inner native environment without changing the outer profile wrapper, action
sampling or context update. Unit tests require exact canonical episode parity.
Native smoke uses one reset, deterministic/one noise and256 steps, permitting
actual short-survival termination. Lightweight final.pt loads on servers; no
replay loads and no CUDA initialization. Publish scalar/window JSON only.

All three Linux nodes eligible, no preference or pin; no Windows. Read small
logs/JSON locally. Source checkpoints stay or copy between servers only.

## Limitations
One stable source policy as positive control. Cross-profile success would show
feasibility for those episodes; cross-profile failure would not prove impossibility.
This is a frozen controller diagnosis, not adaptation or method superiority.
