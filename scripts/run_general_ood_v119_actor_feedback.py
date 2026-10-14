"""Frozen readout actor-feedback pilot; only scalar results leave the worker."""
import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_general_ood_v118_frozen_classifier_confirmation as v118
from scripts import probe_general_ood_v53_behavior_coverage as v53
from scripts.v52_agent import ContinuationAgent

REVISION = "general_ood_v119_actor_feedback"
SEEDS, SMOKE_SEED = (135270, 135271, 135272), 135269
ARMS = ("matched_classifier", "incumbent_v98", "fixed-minus2", "fixed-minus3")
SPEC = dict(seeds=list(SEEDS), steps=6000, switch_step=3800, evaluation_episodes=3, evaluation_max_steps=1000)
CHECKPOINT_STEPS = 1000


class FeedbackReadout:
    def __init__(self, scorer, phase):
        from scripts.v73_recursive_mode_readout import RecursiveModeFilter, original
        self.scorer, self.filter = scorer, RecursiveModeFilter()
        self.departure = original.PhaseDepartureReadout(phase)
        self.risk = self.signal = 0.

    def observe(self, state, executed_action, next_state, episode_age):
        from scripts.v73_recursive_mode_readout import original
        query = dict(states=np.asarray([state]), executed_actions=np.asarray([executed_action]),
                     next_states=np.asarray([next_state]))
        self.scores = self.scorer(query)[0]
        self.mode_scores = self.filter.update(self.scores)
        _, _, signal = original.mode_evidence(self.mode_scores)
        self.signal = float(signal)
        self.risk = float(self.departure.update(self.signal, episode_age))
        return self.risk

    def state(self):
        return dict(filter=self.filter, departure=self.departure, risk=self.risk, signal=self.signal)

    def restore(self, state):
        for key, value in state.items():
            setattr(self, key, value)


class ActorFeedbackAgent(ContinuationAgent):
    @property
    def penalty_coefficient(self):
        if self.variant == "fixed-minus2":
            return 0.
        if self.variant == "fixed-minus3":
            return 1.
        return self.feedback.risk


def cells(seed, arm, smoke=False):
    assert seed == SMOKE_SEED if smoke else seed in SEEDS
    assert arm in ARMS or smoke and arm == "all"
    return [dict(seed=seed, arm=a, mode=m) for a in (ARMS if arm == "all" else (arm,)) for m in v118.v117.MODES]


def windows(spec):
    n, s = spec["steps"], spec["switch_step"]
    return dict(full=(0, n), prefix=(0, s), post=(s, n), first_post50=(s, min(s+50, n)),
                last1000=(max(s, n-1000), n))


def collect(agent, env, cell, spec, apply_mode, loop=None, checkpoint=None):
    from scripts.train_csbapr import _recent_rollout_payload

    if loop is None:
        random.seed(cell["seed"])
        np.random.seed(cell["seed"])
        torch.manual_seed(cell["seed"])
        apply_mode(env, "normal")
        state, _ = env.reset(seed=cell["seed"])
        agent.reset_context()
        loop = dict(step=0, state=state, episode=0, episode_age=0, observer=None,
            rows={k: [] for k in (*v53.TRACE_KEYS,
                "risk", "effective_beta", "q_loss", "scores", "mode_scores", "signal", "threshold")},
            recent={k: deque(maxlen=agent.config.surprise_window) for k in ("state", "action", "reward", "context")})
    elif agent.feedback is not None:
        agent.feedback.restore(loop["observer"])
    for step in range(loop["step"], spec["steps"]):
        if step == spec["switch_step"]:
            apply_mode(env, cell["mode"])
        state = loop["state"]
        action = agent.select_action(state, deterministic=False)
        context = agent._current_context_np().copy()
        executed = np.clip(action*env.action_space.high, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(executed)
        done = terminated or truncated
        # This transition changes the following optimizer step, never its own action.
        if agent.feedback is not None:
            agent.feedback.observe(state, executed, next_state, loop["episode_age"])
        for key, value in (("state", state.copy()), ("action", action.copy()), ("reward", float(reward)), ("context", context)):
            loop["recent"][key].append(value)
        agent.store_transition(state, action, reward, next_state, float(done))
        metrics = agent.update(recent_rollout=_recent_rollout_payload(loop["recent"]))
        coefficient = agent.penalty_coefficient
        assert metrics is not None and abs(metrics["effective_beta"]+2.+coefficient) < 1e-12
        values = dict(states=state.copy(), actions=action.copy(), executed_actions=executed.copy(),
            next_states=next_state.copy(), rewards=float(reward), dones=bool(done), terminated=bool(terminated),
            truncated=bool(truncated), risk=coefficient, effective_beta=float(metrics["effective_beta"]),
            q_loss=float(metrics["q_loss"]), scores=agent.feedback.scores.copy() if agent.feedback else np.zeros(5),
            mode_scores=agent.feedback.mode_scores.copy() if agent.feedback else np.zeros(5),
            signal=agent.feedback.signal if agent.feedback else 0.,
            threshold=float(agent.feedback.departure.departure.threshold) if agent.feedback else 0.)
        assert all(np.isfinite(value).all() for value in values.values()), "nonfinite-feedback-transition"
        for key, value in values.items():
            loop["rows"][key].append(value)
        loop["step"], loop["state"] = step+1, next_state
        loop["episode_age"] += 1
        if done and step+1 < spec["steps"]:
            loop["episode"] += 1
            loop["episode_age"] = 0
            loop["state"], _ = env.reset(seed=cell["seed"]+10000*loop["episode"])
            agent.reset_context()
        if checkpoint and ((step+1) % CHECKPOINT_STEPS == 0 or step+1 == spec["steps"]):
            loop["observer"] = deepcopy(agent.feedback.state()) if agent.feedback else None
            checkpoint(loop)
            print(f"[TRAIN] {cell['arm']}/{cell['mode']} {step+1}/{spec['steps']} beta={metrics['effective_beta']:.5f}", flush=True)
    assert agent.training_steps == 99745+spec["steps"] and agent.replay_buffer.size == 6144+spec["steps"]
    trace = {k: np.asarray(v) for k, v in loop["rows"].items()}
    stats = {w: dict(mean_reward=float(trace["rewards"][a:b].mean()),
        reward_sum=float(trace["rewards"][a:b].sum()), mean_risk=float(trace["risk"][a:b].mean()),
        mean_beta=float(trace["effective_beta"][a:b].mean()), mean_q_loss=float(trace["q_loss"][a:b].mean()),
        physical_terminations=int(trace["terminated"][a:b].sum()), time_limit_truncations=int(trace["truncated"][a:b].sum()))
        for w, (a, b) in windows(spec).items()}
    return trace, dict(**cell, optimizer_updates=spec["steps"], actual_transitions=spec["steps"], windows=stats,
        first_transition={k: trace[k][0].tolist() for k in ("states", "actions", "executed_actions", "next_states", "rewards", "dones")})


def evaluate_final(agent, cell, spec):
    import gymnasium as gym
    from scripts import probe_general_ood_v70_physical_audit as physical
    from scripts.train_csbapr import _evaluate_agent

    context, updates = deepcopy(agent.snapshot_context()), agent.training_steps
    results = {}
    for mode in dict.fromkeys(("normal", cell["mode"])):
        env = gym.make("Hopper-v4")
        physical.apply_mode(env, mode)
        seed = 140000+1000*SEEDS.index(cell["seed"]) if cell["seed"] in SEEDS else 139000
        seed += 100*v118.v117.MODES.index(mode)
        rewards, lengths = [], []
        for i in range(spec["evaluation_episodes"]):
            agent.restore_context(deepcopy(context))
            result = _evaluate_agent(agent, env, spec["evaluation_max_steps"], 1, seed+i)
            rewards.extend(result["rewards"])
            lengths.extend(result["lengths"])
        env.close()
        results[mode] = dict(seed_start=seed, rewards=rewards, lengths=lengths, mean_return=float(np.mean(rewards)))
    agent.restore_context(context)
    assert agent.training_steps == updates
    return results


def make_feedback(arm, family, mean, classifiers, references, incumbent):
    if arm.startswith("fixed-"):
        return None
    if arm == "matched_classifier":
        scorer = lambda query: v118.v117.transition_scores(family.bank, mean, classifiers, query)[arm]
        phase = references[arm].phase
    else:
        scorer = lambda query: family.incumbent.terms(query)[0][v118.PRIMARY]["likelihoods"]
        phase = incumbent[v118.PRIMARY].phase
    return FeedbackReadout(scorer, phase)


def verify_online(trace, reference, expected):
    from scripts.v73_recursive_mode_readout import observe_likelihoods

    np.testing.assert_allclose(trace["scores"], expected, rtol=1e-5, atol=2e-4)
    replay = observe_likelihoods(trace["scores"], trace["dones"], reference, "recursive")
    for key in ("mode_scores", "risk", "signal", "threshold"):
        np.testing.assert_allclose(trace[key], replay[key], rtol=1e-12, atol=1e-12)


def collect_paths(args, source, population, spec, shared, feedback_factory):
    import gymnasium as gym

    helper = source.v76.v68.v53
    extra = torch.load(args.resume_from, map_location="cpu", weights_only=False)["extra_state"] if args.resume_from else None
    progress = extra["progress"] if extra else dict(revision=REVISION, spec=spec, cells=population, next_cell=0, records=[])
    assert progress["revision"] == REVISION and progress["spec"] == spec and progress["cells"] == population
    assert progress["next_cell"] == len(progress["records"])
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir/"latest.pt"
    for i in range(progress["next_cell"], len(population)):
        cell = population[i]
        env = gym.make("Hopper-v4")
        agent = ActorFeedbackAgent(env, cell["arm"], args.device)
        agent.start_shared(args.source_dir/source.v76.v69.SOURCE_NAME, shared)
        agent.feedback = feedback_factory(cell["arm"])
        assert (agent.training_steps, agent.context_tracker._count, agent.replay_buffer.size) == (99745, 100000, 6144)
        loop = None
        if extra and extra["loop"] is not None:
            agent.load(str(args.resume_from))
            loop = extra["loop"]
            source.v76.v70.apply_mode(env, cell["mode"] if loop["step"] > spec["switch_step"] else "normal")
            env.reset(seed=cell["seed"])
            source.v76.v69.restore_step_state(env, extra["environment"])
        trace, record = collect(agent, env, cell, spec, source.v76.v70.apply_mode, loop,
            lambda active: helper.save_progress(agent, env, progress, active, checkpoint))
        path = args.checkpoint_dir/f"{cell['arm']}_{cell['mode']}.npz"
        np.savez_compressed(path, **trace)
        if cell["mode"] != "normal":
            with np.load(args.checkpoint_dir/f"{cell['arm']}_normal.npz") as normal:
                assert helper.paired_prefix_equal(trace, normal, spec["switch_step"]), "nonmatched-within-arm-prefix"
        # Evaluate the saved final actor rather than selecting an intermediate one.
        agent.load(str(checkpoint))
        record["evaluation"] = evaluate_final(agent, cell, spec)
        record["evaluation_interactions"] = sum(sum(r["lengths"]) for r in record["evaluation"].values())
        progress["records"].append(record)
        progress["next_cell"] = i+1
        helper.save_progress(agent, env, progress, None, checkpoint)
        env.close()
        del agent
        torch.cuda.empty_cache()
        extra = None
        print(f"[PATH] {i+1}/{len(population)} {cell['arm']}/{cell['mode']}", flush=True)
    return progress["records"]


def run(args):
    sys.path.insert(0, str(args.v83_dir))
    from scripts import analyze_general_ood_v83_cross_seed_covariance as source

    torch.set_num_threads(2)
    population = cells(args.seed, args.arm, args.smoke)
    spec = dict(seeds=[SMOKE_SEED], steps=64, switch_step=32, evaluation_episodes=1, evaluation_max_steps=64) if args.smoke else SPEC
    family, mean, classifiers, references, incumbent, development = v118.load_readouts(args, source)
    used = {r["seed"] for r in development["records"]} | set(map(int, family.mean_groups)) | set(v118.SEEDS) | {v118.SMOKE_SEED}
    assert not (set(SEEDS) | {SMOKE_SEED}) & used
    metadata = v118.reference_metadata(references, incumbent)
    shared = torch.load(args.shared_dir/"shared.pt", map_location="cpu", weights_only=False)
    records = collect_paths(args, source, population, spec, shared,
        lambda arm: make_feedback(arm, family, mean, classifiers, references, incumbent))
    comparisons = 0
    if args.smoke:
        for cell in population:
            if cell["arm"].startswith("fixed-"):
                continue
            trace = source.v76.v68.load_trace(args.checkpoint_dir/f"{cell['arm']}_{cell['mode']}.npz")
            scores, old = v118.score_trace(family, mean, classifiers, trace)
            ref, expected = ((references[cell["arm"]], scores[cell["arm"]]) if cell["arm"] == "matched_classifier"
                             else (incumbent[v118.PRIMARY], old[v118.PRIMARY]))
            verify_online(trace, ref, expected)
            comparisons += 1
        assert all(r["first_transition"] == records[0]["first_transition"] for r in records)
    assert metadata == v118.reference_metadata(references, incumbent)
    result = dict(revision=REVISION, passed=True, smoke=args.smoke, seed=args.seed, arm=args.arm, spec=spec, records=records,
        actual_transitions=sum(r["actual_transitions"] for r in records), optimizer_updates=sum(r["optimizer_updates"] for r in records),
        evaluation_interactions=sum(r["evaluation_interactions"] for r in records), checked_online_replays=comparisons,
        reference_metadata=metadata, reference_metadata_unchanged=True, reference_refitted=False,
        normal_thresholds_recalibrated=False, readout_controls_actor=True, penalty_enters_actor_loss_only=True,
        beta_mapping="-2-risk; fixed controls -2 and -3; no gain tuning",
        feedback_timing="observe real transition, then same-step optimizer update; affects subsequent actions only",
        within_arm_mode_prefixes_paired=True, cross_arm_first_transition_equal=args.smoke,
        cross_arm_post_feedback_paths_paired=False, counterfactual_step_calls=0, oracle_observations_used_for_scores=False,
        source_training_seed=1, source_updates=99745, independent_training_seed_confirmation=False,
        checkpoint_selection="final", optimizer_updates_during_evaluation=0, readout_adopted=False,
        decision="native-smoke-not-scientific-evidence" if args.smoke else "closed-loop-pilot-not-adoption",
        limitations=["Three environment seeds conditional on one source policy and known mild Hopper modes.",
            "Mode weights enter a scalar risk penalty, not a mode-conditioned policy; reference budgets differ.",
            "Post-feedback state distributions diverge across arms; no same-trajectory causal claim.",
            "Three deterministic final evaluation episodes per condition; short-budget pilot only."])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    path = args.result_dir/("native_smoke.json" if args.smoke else "actor_feedback.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"[DONE] {REVISION}: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("v117-dir", "v116-dir", "v115-dir", "v113-dir", "v104-dir", "v98-dir", "v97-dir", "v95-dir",
                 "v94-dir", "v83-dir", "v78-dir", "v76-dir", "v74-dir", "v73-dir", "v72-dir", "v71-dir",
                 "v69-dir", "v70-dir", "v61-dir", "v62-dir", "source-dir", "shared-dir", "checkpoint-dir", "result-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", choices=(*ARMS, "all"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
