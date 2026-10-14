# v52: normal learning shift blocks formal training

Preflight t135949 completed four matched-start arms, each6000 interactions and
6000 updates, in273.14 s. Shared normal collection cost18432 interactions;
evaluation cost22489. Four focused tests passed, including exact interrupted
continuation and completed-worker penalty accounting. Config, shared reference,
replay, first transition and mode events match. Only74989 bytes of preflight JSON
returned. Formal30000-step arms were not submitted.

Normal-only t135952 diagnoses the unexpected readout: frozen/updating policies
share the first transition and identical physical parameters. Mean coefficient
is .0107086 with zero updates, versus .9153466 with1000 updates. From step100,
updating-policy50-step bins approach .996--.997. Actor parameter L2 change is
0/1.32534. The updating branch exactly reproduces the preflight fixed-minus2
first1000 signal, coefficient and Q loss; beta stays-2, so the new coefficient
does not cause this effect. This is learning-induced behavior/reference shift,
not an environment change. Diagnostic JSON is19218 bytes; task wall52.21 s.

## Known Issues and Next Step

The frozen-policy normal calibration does not transfer to online learning.
This is a development HOLD, not evidence of improved RL returns. Two diagnostic
launches failed before compute because of staging; the shared package was restored
server-side from full checkpoints and `run_shared` is now excluded from staging.
No raw replay/checkpoint was downloaded, and no returns selected parameters.

Next broaden normal state/action coverage with multiple normal behavior policies,
fit/calibrate separate normal splits, and test untouched learning-normal trajectories
before re-running the four-arm OOD pilot. Preserve physical observations, history,
losses and matched budgets. Keep the v51 frozen-policy result and this negative
training result separate; no manuscript claims were changed.
