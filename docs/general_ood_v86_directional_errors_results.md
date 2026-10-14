# v86 Directional error results

Task **t138437** completed on node007 CPU, exit 0, in **121.31s**. Seven local tests passed. All 130 windows / 7,410 transitions and 1,040 fixed-pair comparisons reproduced v78/v83 statistics. Only **1,650,708 bytes** of scalar JSON returned: `results/general_ood_v86_directional_errors/diagnostic/directional_errors.json`.

- v76 seed135180 early slippery/compound versus normal: **92%** cross the common-covariance mean boundary; **68%/62%** exceed neighbors' directional 95th quantile. Their tails have **0%** boundary crossings and query/reference directional second-moment ratios **1.02/0.80**.
- v76 seed135181 heavy tail versus high_grav: mean-position margin **+15.72**, covariance-shape adjustment **-20.00**, volume **+2.57**, total **-1.71**. Directional second-moment ratio is **20.85**; v62 seed134021 heavy reaches **51.28**. The successful v76 seed135180 heavy tail also has ratio **7.57** and 98.44% recursive accuracy.
- v61 slippery tail versus compound: energy **+0.620**, volume **-0.832**, total **-0.212**; volume flips 56.25% of transition margins. Its second-moment ratio is only **1.14**.

Decision: **HOLD, retain v78.** Next: a reference-only shared-covariance pilot, pooling the five frozen v83 covariances equally at each input while retaining means. This removes candidate-specific shape/volume differences. Keep the improved v61 heavy and normal controls; early mean-direction failures require separate repair. No actor/environment/paper changes.

## Limitations
Fixed-pair averages differ from best-other or recursive margins. The true-covariance mean term uses offline labels, not a deployable rule. Reference error transport and development-path diagnostics do not establish a unique causal fix or RL return gains.
