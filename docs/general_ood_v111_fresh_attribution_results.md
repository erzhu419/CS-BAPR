# v111: mean repair alone cannot resolve the fresh failures

Smoke `t141489` and formal `t141490` completed, exit 0. Formal took 64 seconds:
14 windows / 700 existing transitions, 84 component-likelihood comparisons
and 126 instantaneous-summary comparisons passed. Only 210 KB scalar JSON
returned; no collection, calibration or actor updates occurred.

Primary condition: `suffix_disagreement_mix`. All numbers below are
instantaneous accuracy, not the complete-history accuracy reported in v110.

| Updating early window | Normal mixture | Correct simulator centers, old covariance |
| --- | ---: | ---: |
| seed135251 slippery | 32% | 92% |
| seed135252 normal | 44% | 100% |
| seed135252 slippery | 0% | 4% |
| seed135252 high_grav | 6% | 8% |
| seed135252 compound | 0% | 4% |

All four matched frozen windows and six normal controls reach 100% with correct
centers. Seed135252 normal prediction RMS improves 0.45264 to 0.20893, yet its
heavy log density gains 1.59987 versus only 0.52920 for mixed normal density;
the normal-heavy margin falls 1.07066. Lower true-mode RMS is not sufficient.
For its slippery/high_grav/compound correct-center intervention, energy
advantages +2.71/+1.14/+2.22 are outweighed by covariance-volume advantages
-4.48/-2.46/-4.33. The other covariance conditions retain these failures.

**Next:** retain v98 and HOLD actor integration. Pilot source-seed-held-out
prediction-error covariance matched to the expanded mean, keeping raw support,
mixture coefficient and temporal rules fixed. Inspect these windows and the
earlier development failures before another full replay. Do not treat the old
v87 shared-covariance negative result as a successful solution.

**Limitations:** correct simulator centers are diagnostic, not deployable or
a Bayes-optimal ceiling. These failure-selected v110 windows are development
evidence for any redesign; covariance mismatch is a hypothesis to test, not an
established repair. No new repair or training task has been dispatched.
