# v148 within-support frozen recovery diagnostic

Use the four predeclared600k v147 SAC ZIPs, with zero new training/updates and
no replay or GPU initialization. Add only training-known2.0x gravity pulses
lasting10 steps after100/300 completed native steps. Normal gravity returns
without state reset. Native rewards, healthy termination and motor limits
remain unchanged; no labels or onset clocks are policy inputs.

Resets154000--154004, Gaussian noise137000--137002 plus deterministic actions,
native learned standard deviation and1,000-step cap match v147 exactly.
Each model evaluates five conditions in one CPU run: no pulse and ten-step
pulses after100/300 steps at2.0x and original2.659x gravity. This gives400
CPU episodes and320 exact paired pre-pulse prefixes. Native smoke attempts
t152021/t152022 rejected machine-roundoff differences under strict cross-host
equality: observation difference0, qvel difference2.0e-14 and return
difference8.5e-14 in the inspected seed5 deterministic100-step prefix. This
is not invalid v147 evidence. Fresh same-run
controls preserve shared observation/return/qpos/qvel before each intervention.

Report full healthy survival,100-step recovery, delayed falls and pre-exposure
failures separately for each seed/arm/onset. If training-known pulses already
fail, target policy/recovery learning stability; if they succeed but2.659x
fails, target amplitude extrapolation. Native smokes gate formal tasks.

Allow available node007/jtl110gpu2 with no preferred node/GPU or hard pin.
jtl110gpu remains unavailable after its verified root ENOSPC failure; no
cleanup is performed. Small JSON only comes locally; finalZIPs transfer
server-to-server, never replay/full-state checkpoints.
