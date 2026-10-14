# v68 result and next step

Task t136896 completed on node007 CPU in 114.4 s, exit 0. Six frozen-runtime
tests passed. Collected 66,000 fresh reference transitions with zero policy
updates; replayed 102,000 saved transitions. Only the 285 KB scalar JSON was
recovered locally. v66 is retained; v68 is HOLD and is not connected to belief.

| Group / observer | Normal null | Positive first50 | Positive last64 | Positive true-mode margin first50 / tail |
| --- | --- | --- | --- | --- |
| v62 / v66 | pass | 7/8 | 8/8 | - |
| v62 / v68 | fail | 2/8 | 5/8 | 3/8, 4/8 |
| v61 development / v66 | pass | 3/4 | 4/4 | - |
| v61 development / v68 | fail | 2/4 | 4/4 | 2/4, 2/4 |

The normal-null reference has 92% normal-mode accuracy in its first800.
Frozen normal seed134021 instead has risk 0.840 and 87.4% slippery predictions
there; mean normal residual energy rises from 7.91 to 3256.83. Slippery's
energy is 2615.41, overwhelming its 7.65-nat log-determinant penalty. Updating
normal seeds134020/134021 have 96.0%/96.5% slippery predictions in first800.
Thus policy updates alone cannot explain the failure. The evidence points
to prediction/coverage error being interpreted as a physical-mode change.

Next: collect independent common-state counterfactual references, applying
every mode to the same complete simulator state and executed action, with
frozen/updating normal coverage. Verify exact normal one-step replay first,
then test mode separation under the unchanged gates. Do not reconstruct
clipped simulator velocities from observations or fit the old test traces.

## Limitations

Labelled mode reference access is additional prior information. Separate-mode
rollouts do not isolate mode effects from state coverage. Reused test traces
and one source training seed do not establish independent or RL-return gains.
