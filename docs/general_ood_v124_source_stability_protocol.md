# v124 frozen source stability diagnosis

Use every v122 100k and v123 300k final source (training seeds 0/1/2), with
unchanged parameters, replay and source selection. No optimization or readout fit.

For each model, evaluate reset seeds 150000--150004 with deterministic actions
and sampled actions under policy-noise seeds 12400/12401/12402: 20 episodes/model,
120 episodes total, at most 120000 diagnostic interactions. All inference uses
the same native CPU runtime. Original GPU evaluation numbers remain reference
data, not an exact CPU-reproduction requirement. Noise repeats are not new
training seeds. The 1000-step environment time limit and native rewards remain.

Record episode return, velocity, length, action-noise scale, native health
conditions and the last five physical states/actions. These identify which
termination condition was crossed, not the underlying causal dynamics of a fall.

On 256 evenly spaced full-replay and recent-10k samples per model, use the
saved online contexts and three common noise seeds to decompose Q, beta=-2 risk,
entropy and behavior-cloning actor gradients. Verify their sum against the full
loss gradient. Report raw pre-clip gradients and a beta=0 counterfactual gradient;
this is not a trained beta=0 ablation or a measurement of historical Adam updates.

Checkpoints/replay stay native, diagnostic progress is resumable small JSON, and
only scalar JSON returns locally. CPU jobs use two cores; node007 eligibility
reflects native source inputs and simulator parity, with no node/GPU preference
and no Windows placement. Smoke: seed 0, both sources, one reset/noise, 64 steps.
