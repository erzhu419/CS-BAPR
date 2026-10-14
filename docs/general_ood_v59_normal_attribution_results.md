# v59: normal learning reproduces late risk

t136532/t136533 completed with exit0. Four cells match v58 initial transitions
and frozen references; physical parameters stayed normal. All250-step bins and
full/last1000 summaries agree:24000 interactions,12000 updates;89222 scalar bytes.

| Seed | Branch | Phase first800 | Phase last1000 | Phase last64 | Actor L2 change |
| --- | --- | ---: | ---: | ---: | ---: |
|116000|frozen|0.157086|0.032767|0.013920|0|
|116000|updating|0.005798|0.452101|0.799182|4.651039|
|116001|frozen|0.134183|0.136704|0.080949|0|
|116001|updating|0.003518|0.421012|0.812321|4.488388|

OOD exposure is not necessary for the late risk: normal-only learning produces
roughly the v58 late-normal level. Pooled coefficients also rise to0.436561/
0.413181 in the updating branches. Frozen normal windows are not uniformly
clean either. The short v57 calibration does not cover this learning lifetime;
neither changing only reset phase nor blaming adaptive feedback solves it.

## Limitations
One source-training seed and two paired development continuations. Updates also
change stochastic sampling paths, so this does not isolate actor parameters from
all other learning effects. No adaptive RL return improvement is established.

## Next
v60 long-normal coverage preflight: separate6000-step fitting/calibration/test
seeds, old v57 versus long-calibration-only versus long-normal model/refits.
Test both normal branches and paired late changes at3800 under fixed beta=-2.
Require a usable normal null without erasing early/tail change contrasts before
another closed-loop run. No environment, beta or manuscript-claim changes.
