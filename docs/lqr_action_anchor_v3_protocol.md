# LQR action_anchor_v3: frozen isolated ablation

Frozen on 2026-09-30 before formal dispatch. The only algorithm change from CS-BAPR feedback_v2 is a deterministic-action anchor in the actor objective: `1.0 * mean((tanh(actor_mean(state)) - fitted_feedback(state))**2)`, averaged over replay states and action components. The fitted reference is detached. No extra anchor states, pretraining, gradient balancing, critic-scale correction or OOD supervision is added.

- Fresh methods: `csbapr-action-anchor`, `csbapr`, `bapr`, `resac`; seeds 0/1/2, 500 episodes each. Twelve runs; no additional seeds or coefficient search.
- Environment unchanged: fixed 2D double-integrator dynamics, amplitude-1 initial states, normalized actions in [-1, 1], physical force equal to 5 times action, 200 steps per episode.
- Shared training budget unchanged: 1,000 random warmup transitions, then one update per transition; batch 256, hidden width 64, three critics, actor/critic learning rates 0.0003. All stabilization and critic penalties retain their feedback_v2 values.
- Symbolic methods identify controlled degree-1 SINDy on separate amplitude-1 trajectories. The same fitted-feedback construction, action cost and Jacobian weights are retained. Identification is followed by reseeding; the fresh anchored/control pair starts with matched random streams.
- Selection: final episode-500 checkpoint only. Full-state episode-boundary resume, every 20 episodes; one atomic `last.pt` per run. No feedback_v2 checkpoint is used to initialize training.
- Evaluation unchanged: deterministic actions at amplitudes 1/2/5/10/20/50; 20 paired episodes at seeds 10000--10019 per amplitude. Retain all individual returns and failures.
- Also report zero-state action and initial-state action/Jacobian mismatch. Fitted feedback is used for symbolic arms; known-model feedback is diagnostic only for the two baselines. Evaluation does not feed into training or checkpoint selection.
- Primary contrast: anchored CS-BAPR minus contemporaneous unanchored CS-BAPR at every amplitude, including ID. Also compare all seeds against contemporaneous BAPR and RE-SAC. No post-result tuning; no statistical-significance claim from three training seeds.
- Scheduler: no preferred/required node or GPU. Stage a frozen runtime copy and the existing offline scientific wheels; record actual library/GPU versions. Checkpoints stay on workers; synchronize only small evaluation JSON.

## Limitations

This is a task-specific model-derived teacher, not an oracle available in arbitrary RL tasks. The study tests initial-state extrapolation, not changing bus demand. The shared critic target-scale issue is deliberately unchanged for this isolated comparison. BAPR and RE-SAC retain inherited project stabilization; different available CUDA/PyTorch workers remain possible. Prior feedback_v2 evidence and manuscript tables are unchanged.
