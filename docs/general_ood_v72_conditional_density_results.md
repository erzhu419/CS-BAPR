# v72: conditional covariance fixes frozen-path tail identification

t136929 completed on node007 CPU, exit 0, 954.74 seconds. 36,000 exact
true-mode replays, 216,000 simulator calls, zero optimizer updates/new labels.
Five tests passed. Only the 274,661-byte scalar JSON was pulled.

Fresh gravity last64 accuracy: 100%/100% for all three distributions. Fresh
compound: global 0%/0%, conditional-center 0%/26.5625%, conditional-density
100%/100%. Conditional-density also classified all v70/v71 gravity/compound
tails correctly. Its largest gated normal mean across groups was 0.01659.

Overall HOLD: fresh first50 true-mode margins passed 3/4, not 4/4. For compound
seed135131, raw first50 accuracy was 92%, mean margin +10.259; retaining normal
pre-change history gave 76%, mean margin -7.283, with offsets 0-11 all wrong.
Raw transient errors remain, including a -74.857 margin on seed135130.
Old v62/v61 tails passed true-mode margins only 4/8 and 2/4; heavy/slippery
misidentification remains separate from the new frozen-path history delay.

Next: freeze v72 densities and compare the fixed 16-step readout with a causal
recursive mode posterior using a switching prior. Preserve null-calibration
rules and current gates; no evaluation-label tuning or actor feedback. Include
fresh frozen heavy/slippery probes to separate regime coverage from actor drift.

## Limitations

One source seed, mild known profiles and privileged offline simulator labels;
no control-return evidence. Scalar sync initially reported a transport error;
the complete JSON parsed and a single-file rsync retry succeeded.
