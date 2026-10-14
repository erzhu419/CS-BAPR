# v38 belief-response diagnostic

Frozen 2026-10-05. v37 remains a complete negative transfer result. This is a
development screen of the belief mechanism, not a submission confirmation.

## Repair

The legacy likelihood assigns larger variance to longer runs. For a common
prior, greater surprise therefore lengthens the window and weakens the policy
penalty. The corrected filter moves the hazard transition before conditioning
on the current surprise and preserves mature-run mass at the horizon cap.

Let h be the capped run age in {0,...,H-1}, rho the previous run-age
probabilities, q the reset hazard, and xi the current nonnegative surprise.
First form u[0]=q*sum(rho), u[h]=(1-q)*rho[h-1] for h>0, and add
(1-q)*rho[H-1] to the last bin. Its last bin represents all ages >= H-1.
Condition on a half-normal density with variance v[h]=b/(1+k*h), where
b is base_variance and k is variance_growth:
rho_new[h] is proportional to u[h]/sqrt(v[h])*exp(-xi^2/(2*v[h])).
H=20, q=0.02, b=1 and k=0.05 are unchanged. The policy still uses
lambda=sum(rho_new[h]*exp(-0.1*h)) and beta=-2-penalty_scale*lambda after
the existing 100-update warmup. Log-space normalization prevents large
Q-std spikes from underflowing into a uniform posterior.

The reset/growth construction follows run-length filtering, with an explicit
reset-before-observation convention. The fixed half-normal variance schedule
is this diagnostic's modeling choice, not the full parameter-inference model
in [Adams and MacKay (2007)](https://arxiv.org/abs/0710.3742). Existing abstract
Bayes-update proofs are not an RL performance or model-correctness guarantee.

## Matched Matrix

All arms use the unchanged MLP `bapr-stable-context` preset. Legacy and
corrected differ only in the tracker. The fixed-penalty arm uses the corrected
tracker but sets penalty_scale=0: its policy coefficient remains -2, while
context, learning rates, BC, critic losses and entropy settings stay matched.
It is a matched RE-SAC-style control, not v37's differently configured RE-SAC.
NAU, SINDy and Jacobian losses are absent; actor/context ablations wait until
this mechanism has been isolated.

Environments are Pendulum-v1 and Hopper-v4; training seeds are 0 and 1 (12
formal cells). Exact training budgets are respectively 100000 and 300000
interactions. A final partial episode is not marked terminal by the budget
cutoff. Built-in moderate physics modes switch every 5000 interactions using
a separate seed-matched RNG, independent of episode termination. No structured
50x profile is used during training. The final checkpoint alone is selected.
The three arms for each environment/seed are one scheduler bundle with three
concurrent workers on the same Python/library/GPU runtime. Four formal bundles
contain 12 cells. The scheduler chooses an idle compatible Linux node without
preferred or required nodes; different seeds' runtime versions are recorded.

Evaluation uses 20 paired seeds starting at 80000, ID and structured Markov
50x OOD, with switch probability 1 and periods 50/200 for Pendulum/Hopper.
Every non-ID fixed profile gets five episodes. Evaluation performs no optimizer
or belief updates; transition context alone continues online, as in v37.
Inference snapshots now preserve training context normalizers without replay.
Each ID/OOD/fixed-profile block starts from that same training context snapshot,
preventing one evaluation block from changing the next block's input scale.
v37's plain checkpoints omitted these normalizers; its measured scores remain
historical fresh-normalizer outcomes, not a definitive transfer verdict.
Diagnostics record surprise, reset probability, window length, effective
penalty, Q-std and losses in 10000-step blocks. Checkpoints remain remote.

## Decision And Limits

Compare corrected-minus-legacy and corrected-minus-fixed seed-level ID/OOD
returns, episode lengths and all fixed profiles. Retain adverse seeds. Admit
further confirmation only if corrected OOD improves over both controls on both
seeds in at least one environment, with no corresponding all-seed ID regression
and no worse fixed-profile floor. Otherwise inspect the recorded mechanism
signals before any new training. Unit-test direction is not RL evidence.
Two seeds and the smaller Hopper budget cannot establish final convergence,
generic transfer, or a symbolic/NAU advantage. Do not pool v37 and v38 scores.
