# v47: second-switch result and next step

Formal task t135925 completed on node007 GPU0 in 498.21 s (8.3 min).
503808 interactions: 24 reference and 240 test rollouts. Preflight t135924 took
90.47 s; nine focused tests passed. References, normal prefixes and formal stay
branches match v46 exactly; all changed/stay pre-switch arrays and belief histories
match. Only 9594941 bytes of formal JSON returned, no trajectories/checkpoints.

Global-robust changed-minus-stay mean penalty in the first 16 second-switch steps:

| Source policy | heavy -> normal | slippery -> normal | heavy -> slippery | slippery -> heavy |
| --- | --- | --- | --- | --- |
| fixed-minus2 | +.3097, 20/20 | +.3386, 20/20 | +.3250, 20/20 | +.3412, 20/20 |
| relative-adaptive | +.1335, 20/20 | +.4280, 20/20 | +.0002, 11/20 | +.3452, 20/20 |

Counts are positive paired differences, not independent training seeds. No branch
ended in these first16 windows. All changed profiles recover to last64 mean ages
18.83--18.99. Relative heavy -> slippery is not wholly blind: its first64 penalty
difference is +.0879 (19/20 positive), but the early response is inconsistent.

## Limitations and next step

Global-robust still has 67 majority resets after the second first16, all at step1536
immediately after fixed-minus2 TimeLimit: 53 changed-branch, 14 slippery-stay events.
Relative-adaptive has none, but every stay rollout terminates in the final256;
late contrasts therefore also include changed physical trajectories and resets.
These are frozen seed1 development results, not RL performance confirmation.

Next: localize and handle the known reset-start observation using normal-only
references, then measure second-switch response within 50 transitions. Keep matched
stay controls; retain both slow/negative results. No additional training seeds or
RL feedback yet. Full readout: results/general_ood_v47_multiple_switch/summary.json.
