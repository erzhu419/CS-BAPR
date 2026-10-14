# v121: normal-only locomotion source pilot

Motivation: v120 found that the v40 source survives for 1000 steps but moves at only
0.041--0.047 m/s. Test source learning before rebuilding any readout.

- Three fresh training seeds: 0, 1, 2. One fixed-minus2 arm, 100000 interactions each.
- Reuse the v39/v40 context-MLP agent and relative surprise computer. No NAU or SINDy.
  Keep the optimizer, replay, entropy, reward normalization and effective beta=-2 unchanged.
- Only training mode changes: unwrapped normal Hopper-v4 physics, no mode switches.
  Preserve original forward reward weight 1, healthy reward 1 and control cost weight 0.001.
- As in v40, stochastic policy actions start at interaction zero; updates start at
  replay size 256. The config's warmup_steps=5000 was not used by the old runner;
  this pilot does not silently introduce random-action warmup.
- Deterministic normal-environment evaluations at steps 0/25000/50000/75000:
  three episodes, seeds 149000--149002, maximum 1000 steps each. Restore training
  context and random-number states afterward; evaluation never updates replay or parameters.
- Evaluate the saved final checkpoint on five episodes, seeds 150000--150004.
  Report episode velocity, length, physical termination, return, control cost and
  fraction of steps with absolute velocity below 0.1 m/s. No best-checkpoint selection.
- Development target: sustained forward locomotion without losing survival, not a
  return around 1000 produced mostly by the healthy reward. Keep all training seeds.
- Checkpoint every 10000 steps with replay, optimizer/RNG, context, recent rollout,
  episode state and MuJoCo integration state. Retain final.pt separately from latest.pt.
  Models stay on the training node; sync only scalar JSON. No Windows or GPU preferences.
- Formal placement requires the old source runtime (torch 2.5.1, NumPy 2.1.3,
  Gymnasium 1.2.0, MuJoCo 3.2.7), currently verified on node007. Other available GPU
  stacks have missing dependencies or different simulator versions; no node/GPU preference.
- Smoke: 512 interactions, checkpoint/diagnostics every 256 steps, evaluation capped
  at 64 steps and one episode; unchanged batch size and model architecture.

Next decision: if normal-only sources acquire useful locomotion, rebuild source-matched
references and rerun the four-arm feedback comparison. If they remain nearly stationary,
isolate exploration and objective/entropy scaling before further readout experiments.
This source-development pilot is not evidence of OOD superiority; v119's negative result stays.
