# v60: normal coverage improved, early response still HOLD

t136535 completed with exit0. All16 cells,96000 interactions/72000 updates,
10 updating paired prefixes, contrasts and gates match; scalar JSON117638 bytes.
Calibration-only leaves the old dynamics, moments and score scale unchanged.

| Sensor | Normal bound | First50 positive pairs | Last64 positive pairs |
| --- | --- | ---: | ---: |
|v57 frozen|fail|7/8|8/8|
|long calibration only|pass|6/8|4/8|
|long normal refit|pass|5/8|8/8|

Refitted updating-normal last1000 coefficients are0.008492/0.017548, versus
0.338467/0.326888 with v57. Refitted tail contrasts are positive in all eight
changes. Only widening thresholds suppresses four tails to exactly zero.

All three refitted early failures occur on132020: slippery-0.007285,
high_grav-0.036603, compound-0.051361. Its no-change first50 coefficient is
0.431029; changed coefficients are0.423744/0.394426/0.379668. Signals are
1.061427 normal versus1.193997/0.829886/0.969407 changed. Thus smoothing and
threshold timing may matter for slippery; gravity/compound also reduce the
scalar signal. Positive distance from a fixed normal center is not necessarily
an increasing measure of every environment change.

## Limitations
The declared wide-window normal bound passes, but a short normal pulse remains
at the change-time window. One source policy, two development test seeds and no
adaptive return confirmation. v60 remains HOLD; the first50 gate is unchanged.

## Next
v61 targeted matched reruns of132020 normal and four changed profiles, with the
frozen v60 refit. Decompose signed residual movement, distance cross terms,
instantaneous risk and16-step smoothing in short post-change windows. Use the
known change boundary only for offline diagnosis, not an actor/controller.
Retain all five traces server-side and return only scalar/11-coordinate summaries.
