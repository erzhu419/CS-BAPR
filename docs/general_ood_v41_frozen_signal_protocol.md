# v41 frozen-policy signal probe

Frozen2026-10-06 after v40 HOLD. Test observation quality before another RL run. Use v40 seed1 final `fixed-minus2` and `relative-adaptive` checkpoints; actor, critic and target critic frozen. Auxiliary surprise/belief calculations never affect actions. No optimizer calls, replay insertions or new training seeds.

## Paired Environment Runs

For each policy,20 evaluation seeds90000-90019. Collect1536 transitions per case. Start in normal mode; before transition1024 either reapply normal (no-change control) or apply one of the four unchanged v40 training profiles: heavy, slippery, high_grav, compound. Five cases per seed,100 per policy,307200 environment interactions overall. These are not structured50x OOD benchmark scores.

Use deterministic actions, restore training context before every rollout, then allow normal online context updates. Reset the physical environment/context EMA after termination, retaining the selected mode and using seed+10000*episode_index. Verify states, actions, stored contexts, rewards, endings and frozen-critic Q spreads are exactly paired before the intervention. No changes to physics profiles or oracle inputs. Set probe TimeLimit to1536 instead of1000 to avoid a scheduled timeout inside the observation interval; physical failure remains active. Record episode endings and reward means per256-transition segment.

## Observers

All observers consume the same collected rewards and Q spreads. Q spreads are computed with frozen weights on each stored state/action/pre-transition-context; global current/target critic L1 norms are fixed. Reward normalization std is1 throughout, explicitly excluding the moving replay normalization of v40. This tests observations/cadence with learning removed, not an exact replay of training surprise.

- `step-overlap`: existing relative surprise on the most recent256 transitions, queried every transition from end256 onward.
- `chunk256`: the identical detector queried once per non-overlapping256-transition chunk.
- `previous-chunk-reference`: at chunk boundaries, reward signal is abs(current mean-previous mean)/max(previous population std,1e-6); Q signal is the existing positive current/previous mean-spread ratio; L1 relative drift is fixed. Use the same max aggregation and surprise EMA. First chunk supplies the reference only; subsequent chunks score against history before updating that reference.

Each auxiliary age filter starts uniform and uses unchanged H=20, hazard=.02 per observation, base variance1, existing likelihood/age cap and decay exp(-.1*h). Report observations, signal components, posterior reset/age/entropy and hypothetical beta=-2-lambda in256-transition bins. No optimizer warmup applies to these diagnostic maps. Hypothetical beta is not applied to the frozen policy. Cadence changes the filter's time unit and initialization history; compare paired actual-minus-control responses, not absolute age/beta across observers as if their time units matched.

## Execution And Interpretation

First CUDA preflight: two separate seeds99000-99001, normal/high_grav, both frozen policies (8 rollouts,12288 transitions). Require valid budgets, frozen source policy counters, no replay writes, paired prefixes, expected observer query counts and finite outputs. Then formal runs use the same frozen source, without selecting windows, coefficients or profiles from preflight scores.

Read checkpoints on node007, where they already reside, rather than downloading or copying them. This is data locality, not machine preference; scheduler chooses an idle GPU there, with no preferred/required GPU. No Windows. Only JSON/logs sync locally; no checkpoints or full trajectories.

Inspect first/second post-switch chunks versus each seed's no-change case, separately by policy and target profile, together with physical failures. Retain weak and negative contrasts. A useful signal should distinguish actual changes from sham behavior across multiple profiles without merely reflecting resets. This is a judgment from paired traces, not a new tuned numerical gate. Signal separation alone does not demonstrate improved RL control; if supported, the next step is a small matched-budget RL test of the revised observation clock/reference.

## Limits

One trained seed, two existing policies,20 probe seeds; not independent training confirmation. Context adapts but model parameters do not. Cadences have different hazard/age time units and few chunk observations. Fixed reward scale and frozen critic exclude two sources of training drift. Termination/reset timing diverges after interventions. Extended probe TimeLimit and training-mode perturbations prevent treating these rollouts as v40 benchmark returns or50x OOD claims.

Post-run clarification: ending counters combine termination/truncation. First-post transitions1024-1279 are before TimeLimit; final-window endings can be timeouts unless a preceding reset makes the1536-step local time limit unreachable. Summary marks this boundary without changing frozen probe source or discarding either post window.
