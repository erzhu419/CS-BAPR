from copy import deepcopy

import pytest

from scripts import summarize_general_ood_v136 as merger
from test_general_ood_v133_summary import make_extension_rows


def make_rows():
    _, originals = make_extension_rows()
    sources = [r for r in originals if r["arm"] == "mse-control"]
    profiles = merger.worker.v135.make_ood_profiles("Hopper-v4", intensity=50., family="structured")
    by_name = {p["name"]: p for p in profiles}
    rows = []
    for source in sources:
        spec = merger.worker.stage_spec(source["training_seed"])
        pairs = []
        for cell in merger.worker.cells(spec):
            early = cell["condition"] == "abrupt/mega_event" and cell["reset_seed"] == 153004 and cell["noise_seed"] == 136002
            target = cell["condition"].removeprefix("abrupt/")
            episodes = []
            for context in merger.worker.CONTEXT_MODES:
                length = 100 if early else ((1000 if context == "online" else 240) if target == "id" else (260 if context == "online" else 250))
                post = max(length-200, 0)
                events = [dict(step=step, profile=name, actual_multipliers={k: by_name[name].get(k, 1.)
                    for k in ("body_mass", "gravity", "friction")}) for step, name in
                    ([(1, "id")]+([(201, target)] if target != "id" and post else []))]
                state = dict(height=1., angle=0., max_abs_qvel=1., unhealthy_conditions=[])
                episodes.append(dict(cell, context_mode=context, length=length, episode_return=float(length),
                    mean_velocity=1., native_reward_verified=True, physical_termination=length < 1000,
                    time_limit_truncation=length == 1000, diagnostic_cutoff=False,
                    noise_scale=0. if cell["mode"] == "deterministic" else 1.,
                    mean_action_noise_l2=0. if cell["mode"] == "deterministic" else .1,
                    warmup_steps=min(length, 200), warmup_return=float(min(length, 200)),
                    intervention_reached=bool(post), post_warmup_steps=post, post_warmup_return=float(post),
                    post_warmup_mean_velocity=1. if post else None,
                    context_updates=length if context == "online" else min(length, 200), mean_context_l2=.5,
                    mean_post_context_change_l2=(.1 if context == "online" else 0.) if post else None,
                    context_at_intervention=[.4]*28 if length >= 200 else None,
                    warmup_end_physical_state=state if length >= 200 else None,
                    profile_counts={"id": 200, target: post} if target != "id" and post else {"id": length},
                    profile_events=events, switches=len(events)-1, non_id_steps=post if target != "id" else 0,
                    final_physical_state=dict(state, unhealthy_conditions=["height"] if length < 1000 else [])))
            pairs.append(dict(cell, episodes=episodes, prefix_equal=True, prefix_steps=min(episodes[0]["length"], 201),
                              prefix_max_abs_difference=0.))
        rows.append(dict(revision=merger.worker.REVISION, training_seed=source["training_seed"], spec=spec, smoke=False,
            profiles=profiles, runtime=source["runtime"], evaluation_device="cpu", cuda_initialized=False,
            replay_loaded=False, optimizer_updates=0, new_training_interactions=0, source_readout_refitted=False,
            checkpoint_selection="predeclared-final-only", reward_changes=[], policy_receives_profile_labels=False,
            ablation_scope="post-warmup inference context, not training representation", seconds=1., pairs=pairs,
            evaluation_interactions=sum(e["length"] for p in pairs for e in p["episodes"]),
            source=dict(revision=source["revision"], checkpoint=source["checkpoint"], context_count=300000,
                training_steps=299745, initial_context_normalizer=dict(count=300000), config=dict(source["config"], device="cpu"))))
    return rows, sources


def test_summary_uses_post_warmup_outcomes_and_retains_early_failures():
    result = merger.summarize(*make_rows())
    assert result["evaluation_episodes"] == 400 and result["verified_prefixes"] == 200
    assert result["independent_training_seeds"] == 2 and result["optimizer_updates"] == 0
    assert all(r["early_failure_episodes"] == 2 for r in result["records"])
    groups = result["records"][0]["groups"]
    assert groups["id/sampled"]["online"]["mean_post_warmup_steps"] == 800.
    assert groups["id/sampled"]["held-context"]["mean_post_warmup_steps"] == 40.
    assert groups["abrupt/mega_event/sampled"]["online"]["intervention_reached"] == 14
    diffs = {(r["condition"], r["mode"]): r for r in result["paired_differences"] if r["training_seed"] == 3}
    assert diffs["id", "sampled"]["online_minus_held"]["post_warmup_steps"] == 760.
    assert diffs["abrupt/heavy_traffic", "sampled"]["online_minus_held"]["post_warmup_steps"] == 10.


@pytest.mark.parametrize("failure", ["missing", "wrong_source", "wrong_noise", "bad_prefix", "changed_warmup", "held_updates", "zero_context", "wrong_switch", "wrong_return"])
def test_rejects_unmatched_interventions_and_protocol(failure):
    rows, sources = deepcopy(make_rows())
    pair = rows[0]["pairs"][60]
    episode = pair["episodes"][1]
    if failure == "missing":
        rows.pop()
    elif failure == "wrong_source":
        rows[0]["source"]["checkpoint"] = "best.pt"
    elif failure == "wrong_noise":
        episode["noise_seed"] = 42
    elif failure == "bad_prefix":
        pair["prefix_max_abs_difference"] = 1.
    elif failure == "changed_warmup":
        episode["warmup_return"] += 1.
    elif failure == "held_updates":
        episode["context_updates"] = episode["length"]
    elif failure == "zero_context":
        episode["mean_context_l2"] = 0.
    elif failure == "wrong_switch":
        episode["profile_events"][-1]["step"] = 200
    else:
        episode["post_warmup_return"] += 1.
    with pytest.raises((AssertionError, ValueError)):
        merger.summarize(rows, sources)
