from copy import deepcopy

import pytest

from scripts import summarize_general_ood_v135 as merger
from test_general_ood_v133_summary import make_extension_rows


def make_rows():
    _, originals = make_extension_rows()
    sources = [r for r in originals if r["arm"] == "mse-control"]
    profiles = merger.worker.make_ood_profiles("Hopper-v4", intensity=50., family="structured")
    by_name = {p["name"]: p for p in profiles}
    rows = []
    for source in sources:
        for context in merger.worker.CONTEXT_MODES:
            seed = source["training_seed"]
            spec = merger.worker.stage_spec(seed, context)
            episodes = []
            for cell in merger.worker.cells(spec):
                condition = cell["condition"]
                length = (1000 if context == "online" else 800) if condition == "id" else (
                    180 if context == "online" else (100 if condition == "markov" else 120))
                names = ["id", "easy_surface", "id", "easy_surface"] if condition == "markov" else [condition.removeprefix("static/")]
                starts = range(1, length+1, 50) if condition == "markov" else [1]
                events = [dict(step=step, profile=names[i], actual_multipliers={
                    k: by_name[names[i]].get(k, 1.) for k in ("body_mass", "gravity", "friction")})
                    for i, step in enumerate(starts)]
                counts = {}
                for i, event in enumerate(events):
                    end = events[i+1]["step"] if i+1 < len(events) else length+1
                    name = event["profile"]
                    counts[name] = counts.get(name, 0)+end-event["step"]
                episodes.append(dict(cell, context_mode=context, length=length, episode_return=float(length),
                    mean_velocity=1., physical_termination=length < 1000, time_limit_truncation=length == 1000,
                    diagnostic_cutoff=False, native_reward_verified=True, noise_scale=0. if cell["mode"] == "deterministic" else 1.,
                    mean_action_noise_l2=0. if cell["mode"] == "deterministic" else .1,
                    context_updates=length if context == "online" else 0, mean_context_l2=.5 if context == "online" else 0.,
                    profile_counts=counts, profile_events=events, switches=len(events)-1,
                    non_id_steps=sum(n for name, n in counts.items() if name != "id")))
            rows.append(dict(revision=merger.worker.REVISION, training_seed=seed, context_mode=context,
                spec=spec, smoke=False, evaluation_device="cpu", cuda_initialized=False, replay_loaded=False,
                optimizer_updates=0, new_training_interactions=0, checkpoint_selection="predeclared-final-only",
                source_readout_refitted=False, reward_changes=[], policy_receives_profile_labels=False,
                ablation_scope="inference context, not training representation", profiles=profiles,
                runtime=source["runtime"], seconds=1., evaluations=episodes,
                evaluation_interactions=sum(e["length"] for e in episodes), source=dict(revision=source["revision"],
                    checkpoint=source["checkpoint"], context_count=300000, training_steps=299745,
                    initial_context_normalizer=dict(count=300000), config=dict(source["config"], device="cpu"))))
    return rows, sources


def test_paired_summary_separates_id_and_ood_and_accepts_different_exposure_lengths():
    result = merger.summarize(*make_rows())
    assert result["independent_training_seeds"] == result["frozen_models"] == 2
    assert result["evaluation_episodes"] == 480
    assert result["new_training_interactions"] == result["optimizer_updates"] == 0
    groups = {(r["training_seed"], r["condition"], r["mode"]): r for r in result["paired_differences"]}
    assert groups[3, "id", "sampled"]["online_minus_zero"]["length"] == 200.
    assert groups[3, "static/mega_event", "sampled"]["online_minus_zero"]["length"] == 60.
    assert groups[3, "markov", "sampled"]["online_minus_zero"]["length"] == 80.
    assert groups[3, "markov", "sampled"]["length_wins"] == 15
    assert result["records"][0]["groups"]["markov/sampled"]["mean_switches"] == 3.
    assert result["records"][1]["groups"]["markov/sampled"]["mean_switches"] == 1.


@pytest.mark.parametrize("failure", ["missing", "wrong_source", "wrong_noise", "zero_updates", "wrong_physics", "wrong_switch", "changed_reward", "nonfinite"])
def test_rejects_unmatched_or_incorrect_protocol(failure):
    rows, sources = deepcopy(make_rows())
    episode = rows[1]["evaluations"][-1]
    if failure == "missing":
        rows.pop()
    elif failure == "wrong_source":
        rows[1]["source"]["checkpoint"] = "best.pt"
    elif failure == "wrong_noise":
        episode["noise_seed"] += 1
    elif failure == "zero_updates":
        episode["context_updates"] = episode["length"]
    elif failure == "wrong_physics":
        episode["profile_events"][-1]["actual_multipliers"]["gravity"] = 50.
    elif failure == "wrong_switch":
        episode["profile_events"][-1]["step"] = 50
    elif failure == "changed_reward":
        episode["native_reward_verified"] = False
    else:
        episode["mean_velocity"] = float("nan")
    with pytest.raises((AssertionError, ValueError)):
        merger.summarize(rows, sources)
