# v48: reject mean-only correction; measure first50 response

Formal task t135932 completed on node007 GPU0 in 594.09 s (9.9 min): 575488
interactions, 24 reference and 280 test rollouts. Preflight t135930 took 89.45 s;
nine focused tests passed. V47 controls and v46 normal overlap match exactly.
Only 7541251 bytes of formal scalar JSON returned, no trajectories/checkpoints.

Mean-only startup correction is unsupported. Fixed-minus2 has 87 late majority
resets in both arms, all at transition1536 after TimeLimit. The new extended normal
control has 20/20 cases with that reset. Same-past correction removes none of them.
Normal calibration startup mean signal barely changes: 5.76974 -> 5.76955 (8
starts); relative-adaptive 3.54429 -> 3.51143 (16 starts), with no late majority resets.

Global-robust first50 paired penalty is positive in 20/20 cases for seven of eight
changed combinations. Relative heavy -> slippery is +.07298, 19/20 positive;
first16 remains +.00024, 11/20. Mean effects are +.0589 in steps21--25 and +.1524
in steps26--30. Seed90019 remains negative over first50 (-.06513). All changed
and stay first50 windows have zero termination/truncation, so endings do not
explain this early-response pattern.

## Limitations and next step

These reuse two frozen seed1 development policies; no RL performance confirmation.
The startup reference is small, and later contrasts include different trajectories
and episode endings. Do not carry the mean-only arm into training.

Next: estimate startup observation uncertainty from normal references, and handle
it in both predictive likelihood and history statistics. Keep the existing global
control, no hard belief reset or observation masking, then repeat the matched
normal/multiple-switch readout before considering RL feedback. Full curves remain
in results/general_ood_v48_reset_center/summary.json and the per-policy scalar JSON.
