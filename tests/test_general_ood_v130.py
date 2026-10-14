from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from scripts import analyze_general_ood_v130_independent_replication as worker
from scripts import summarize_general_ood_v130 as merger
from scripts import run_general_ood_v122_target_l1 as prefix_runner
from scripts import run_general_ood_v123_source_extension as source_runner
from scripts import run_general_ood_v128_mean_retention as continuation_runner


@pytest.mark.parametrize("seed", (3, 4))
@pytest.mark.parametrize("runner", (prefix_runner, source_runner, continuation_runner))
def test_training_cli_accepts_replication_seeds_without_changing_stage(monkeypatch, seed, runner):
    captured = []
    monkeypatch.setattr(runner, "run", lambda args: captured.append(args))
    argv = ["runner", "--seed", str(seed), "--checkpoint-dir", "checkpoints", "--result-dir", "results"]
    if runner is not prefix_runner:
        argv += ["--source-runtime", "source"]
    if runner is continuation_runner:
        argv += ["--arm", "mean-only-source-regularization"]
    monkeypatch.setattr("sys.argv", argv)
    runner.main()
    assert captured[0].seed == seed and captured[0].device == "cuda" and not captured[0].smoke


def episode(cell, length=100):
    return dict(cell, length=length, noise_scale=0. if cell["mode"] == "deterministic" else 1.,
        episode_return=float(length*2), mean_velocity=1., physical_termination=True,
        time_limit_truncation=False, diagnostic_cutoff=False)


def test_common_protocol_and_resume_only_missing_cells(monkeypatch):
    for seed in worker.SEEDS:
        spec = worker.stage_spec(seed)
        assert spec["reset_seeds"] == list(range(151000, 151005))
        assert spec["noise_seeds"] == list(range(14000, 14020)) and spec["max_steps"] == 1000
        assert len(worker.v129.cells(spec)) == 315
    spec = worker.stage_spec(4, True)
    expected = worker.v129.cells(spec)
    assert len(expected) == 6 and all(c["mode"] == "deterministic" for c in expected[:3])
    calls, saves = [], []
    def evaluate(agents, cell, max_steps):
        calls.append(cell)
        return episode(cell)
    monkeypatch.setattr(worker.v129, "evaluate_cell", evaluate)
    progress = dict(evaluations=[episode(c) for c in expected[:4]])
    worker.collect({}, spec, progress, lambda: saves.append(len(progress["evaluations"])))
    assert calls == expected[4:] and saves == [5, 6]
    worker.collect({}, spec, progress, lambda: pytest.fail("completed cells were repeated"))
    progress["evaluations"][0]["reset_seed"] = 150000
    with pytest.raises(AssertionError):
        worker.collect({}, spec, progress, lambda: None)


def small_agent(env, device, beta):
    config = worker.v128.v121.v39.v38.make_config("Hopper-v4", "fixed-penalty", device)
    config.state_dim, config.action_dim = 11, 3
    config.hidden_dim, config.batch_size, config.weight_reg = 16, 8, 0.
    return worker.v128.v125.ActorBetaAgent(11, 3, config, beta)


@pytest.mark.parametrize("seed", (2, 3, 4))
def test_native_source_routing_and_frozen_final_arms(tmp_path, monkeypatch, seed):
    torch.set_num_threads(1)
    monkeypatch.setattr(worker.v128.v125, "make_agent", small_agent)
    args = SimpleNamespace(seed=seed, v123_runtime=tmp_path/"v123", v128_runtime=tmp_path/"v128",
        replication_runtime=tmp_path/"v130")
    entries = worker.source_entries(args)
    assert entries[0][2].is_relative_to(args.v123_runtime if seed < 3 else args.replication_runtime)
    assert entries[1][2].is_relative_to(args.v128_runtime if seed < 3 else args.replication_runtime)
    env = gym.make("Hopper-v4")
    reference_checkpoint = None
    for model, age, path, arm in entries:
        agent = small_agent(env, "cpu", -2.)
        agent.training_steps = age-agent.config.batch_size+1
        agent.context_tracker._count = age
        final = tmp_path/f"{model}.pt"
        agent.save(str(final))
        data = dict(revision=worker.v128.v123.REVISION if arm is None else worker.v128.REVISION,
            training_seed=seed, smoke=False, training=dict(interactions=age), effective_beta=-2.,
            config=asdict(agent.config), checkpoint_selection="final-not-best", source_readout_refitted=False,
            periodic_evaluation_selects_checkpoint=False, runtime={"torch": "test"}, checkpoint=str(final),
            spec=worker.v128.v123.stage_spec() if arm is None else worker.v128.stage_spec(arm))
        if arm is None:
            reference_checkpoint = str(final)
        else:
            data.update(arm=arm, mean_retention_coefficient=worker.v128.ARMS[arm], reference_frozen=True,
                current_std_directly_regularized=False, initial_normalizer_swapped=False,
                reference_checkpoint=reference_checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    env.close()
    agents, sources = worker.load_sources(args, {"torch": "test"})
    assert list(agents) == list(worker.MODELS) and [s["age"] for s in sources] == [300000, 350000, 350000]
    assert all(not p.requires_grad for a in agents.values() for p in a.actor.parameters())
    wrong = json.loads(entries[-1][2].read_text())
    wrong["reference_checkpoint"] = "another-seed.pt"
    entries[-1][2].write_text(json.dumps(wrong))
    with pytest.raises(AssertionError):
        worker.load_sources(args, {"torch": "test"})


def formal_rows():
    rows = []
    for seed in worker.SEEDS:
        spec = worker.stage_spec(seed)
        lengths = dict(zip(worker.MODELS, (200, 100, 90 if seed == 0 else 300)))
        evaluations = [episode(c, lengths[c["model"]]) for c in worker.v129.cells(spec)]
        sources = [dict(model=m, age=age, config={}, revision=revision, arm=arm) for m, age, revision, arm in zip(
            worker.MODELS, (300000, 350000, 350000), (worker.v128.v123.REVISION, worker.v128.REVISION,
                worker.v128.REVISION), (None, *worker.v128.ARMS))]
        rows.append(dict(revision=worker.REVISION, spec=spec, training_seed=seed, evaluation_device="cpu",
            sources=sources, evaluations=evaluations, runtime={"mujoco": "test"},
            optimizer_updates=0, new_training_interactions=0, source_readout_refitted=False,
            checkpoint_selection="predeclared-final-only", reward_or_physics_changes=[],
            evaluation_interactions=sum(e["length"] for e in evaluations)))
    return rows


def test_summary_keeps_five_training_seeds_and_negative_outcomes():
    rows = formal_rows()
    result = merger.summarize(rows)
    assert result["evaluation_episodes"] == 1575 and result["frozen_models"] == 15
    assert result["independent_training_seeds"] == 5 and result["new_training_seeds"] == [3, 4]
    assert len(result["per_noise"]) == 100
    assert [r["independent_replication"] for r in result["per_training_seed"]] == [False]*3+[True]*2
    assert result["per_training_seed"][0]["modes"]["sampled"]["retention_minus_control"]["mean_length"] == -10.
    for wrong in (rows[:-1], rows+[rows[-1]]):
        with pytest.raises(AssertionError):
            merger.summarize(wrong)
    for key, value in (("noise_seed", 13000), ("noise_scale", .5)):
        wrong = deepcopy(rows)
        wrong[-1]["evaluations"][-1][key] = value
        with pytest.raises(AssertionError):
            merger.summarize(wrong)
    wrong = deepcopy(rows)
    wrong[-1]["runtime"]["mujoco"] = "different"
    with pytest.raises(AssertionError):
        merger.summarize(wrong)


def test_training_cost_requires_all_new_paths_and_own_references(tmp_path):
    paths = []
    for seed in (3, 4):
        config = dict(weight_reg=0., batch_size=256)
        prefix = dict(revision=worker.v128.v123.v122.REVISION, training_seed=seed, smoke=False,
            spec=worker.v128.v123.v122.stage_spec(), config=config,
            training=dict(interactions=100000, optimizer_updates=99745))
        source = dict(revision=worker.v128.v123.REVISION, training_seed=seed, smoke=False,
            spec=worker.v128.v123.stage_spec(), config=config, checkpoint=f"seed{seed}/final.pt",
            training=dict(interactions=300000, optimizer_updates=299745),
            actual_new_training_interactions=200000, actual_new_optimizer_updates=200000)
        entries = [(tmp_path/f"prefix_seed{seed}/source.json", prefix),
                   (tmp_path/f"source_seed{seed}/source.json", source)]
        for arm in worker.v128.ARMS:
            entries.append((tmp_path/f"seed{seed}_{arm}/mean_retention.json", dict(revision=worker.v128.REVISION,
                training_seed=seed, smoke=False, arm=arm, spec=worker.v128.stage_spec(arm), config=config,
                reference_checkpoint=source["checkpoint"], checkpoint_selection="final-not-best",
                actual_new_training_interactions=50000, actual_new_optimizer_updates=50000,
                training=dict(interactions=350000, optimizer_updates=349745))))
        for path, data in entries:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            paths.append(path)
    cost = merger.training_cost(tmp_path)
    assert cost["new_training_interactions"] == 800000 and cost["new_optimizer_updates"] == 799490
    wrong = json.loads(paths[-1].read_text())
    wrong["reference_checkpoint"] = "seed0/final.pt"
    paths[-1].write_text(json.dumps(wrong))
    with pytest.raises(AssertionError):
        merger.training_cost(tmp_path)
