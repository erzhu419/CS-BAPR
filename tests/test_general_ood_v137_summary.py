from copy import deepcopy

import pytest

from scripts import summarize_general_ood_v137 as merger


def make_rows():
    rows = []
    for seed in (3, 4):
        for profile in merger.worker.PROFILES:
            spec = merger.worker.stage_spec(profile)
            settings = next(p for p in merger.worker.v135.make_ood_profiles("Hopper-v4", 50., "structured") if p["name"] == profile)
            multipliers = {k: settings.get(k, 1.) for k in ("body_mass", "gravity", "friction")}
            protocol = merger.worker.evaluation_spec()
            length = 80 if profile == "mega_event" else 1000
            condition = "id" if profile == "id" else "static/"+profile
            episodes = [dict(condition=condition, mode="deterministic" if noise is None else "sampled",
                reset_seed=reset, noise_seed=noise, length=length, episode_return=float(length), mean_velocity=1.,
                native_reward_verified=True, context_mode="online", context_updates=length,
                noise_scale=0. if noise is None else 1., physical_termination=length < 1000,
                time_limit_truncation=length == 1000, diagnostic_cutoff=False,
                profile_counts={profile: length}, switches=0, non_id_steps=0 if profile == "id" else length,
                profile_events=[dict(step=1, profile=profile, actual_multipliers=multipliers)])
                for reset in protocol["reset_seeds"] for noise in (None, *protocol["action_noise_seeds"])]
            rows.append(dict(revision=merger.worker.REVISION, training_driver_revision=merger.worker.v132.REVISION,
                training_seed=seed, training_profile=profile, spec=spec, smoke=False, arm="mse-control",
                training_device="cuda", trained_from_scratch=True, training_mode="fixed-profile-from-scratch",
                effective_beta=-2., config=dict(weight_reg=0., use_context=True, use_nau_actor=False,
                    weight_sym=0., jac_weight=0., action_anchor_weight=0., batch_size=256), runtime=dict(gpu=f"gpu{seed}"),
                reward_changes=[], policy_receives_profile_labels=False, checkpoint_selection="final-not-best",
                checkpoint=f"server/seed{seed}_{profile}/final.pt", periodic_evaluation_selects_checkpoint=False,
                source_readout_refitted=False, optimizer_updates_during_evaluation=0, seconds=1.,
                reward_settings=dict(healthy_reward=1., ctrl_cost_weight=.001, forward_reward_weight=1.),
                physics=dict(profile=profile, actual_multipliers=multipliers), final_context_count=300000,
                training=dict(interactions=300000, optimizer_updates=299745,
                    diagnostic_blocks=[dict(step=step, interactions=5000, updates=4745 if step == 5000 else 5000)
                        for step in range(5000, 300001, 5000)],
                    periodic_evaluations=[dict(step=step, profile=profile, actual_multipliers=multipliers,
                        episodes=[dict(seed=reset) for reset in range(149000, 149003)]) for step in range(0, 300000, 25000)]),
                final_evaluation=dict(protocol=protocol, device="cpu", profile=profile, episodes=episodes,
                    optimizer_updates=0, replay_loaded=False, evaluation_interactions=20*length)))
    return rows


def test_summary_preserves_negative_profiles_and_training_seed_unit():
    result = merger.summarize(make_rows())
    assert result["training_interactions"] == 1800000 and result["optimizer_updates"] == 1798470
    assert result["final_evaluation_episodes"] == 120 and result["independent_training_seeds"] == 2
    assert result["records"][0]["modes"]["sampled"]["time_limits"] == 15
    assert result["records"][2]["modes"]["sampled"]["falls"] == 15


@pytest.mark.parametrize("failure", ["missing", "wrong_budget", "wrong_mask", "wrong_periodic_profile", "wrong_final_profile", "wrong_physics", "wrong_noise", "updates"])
def test_rejects_unmatched_budget_or_profile(failure):
    rows = deepcopy(make_rows())
    row = rows[2]
    if failure == "missing":
        rows.pop()
    elif failure == "wrong_budget":
        row["training"]["interactions"] -= 1
    elif failure == "wrong_mask":
        row["spec"]["bootstrap_mask"] = "terminated-or-truncated"
    elif failure == "wrong_periodic_profile":
        row["training"]["periodic_evaluations"][0]["profile"] = "id"
    elif failure == "wrong_final_profile":
        row["final_evaluation"]["episodes"][0]["condition"] = "id"
    elif failure == "wrong_physics":
        row["physics"]["actual_multipliers"] = dict(body_mass=50., gravity=50., friction=50.)
    elif failure == "wrong_noise":
        row["final_evaluation"]["episodes"][-1]["noise_seed"] += 1
    else:
        row["final_evaluation"]["optimizer_updates"] = 1
    with pytest.raises((AssertionError, ValueError)):
        merger.summarize(rows)
