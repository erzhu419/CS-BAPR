"""Matched known-profile learnability, keeping training seeds separate."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v137_fixed_profile as worker
from summarize_general_ood_v129 import aggregate


def validate(row, smoke=False, study=worker):
    json.dumps(row, allow_nan=False)
    seed, profile = row["training_seed"], row["training_profile"]
    spec = study.stage_spec(profile, smoke)
    assert seed in (3, 4) and row["revision"] == study.REVISION and row["smoke"] is smoke
    assert row["training_driver_revision"] == study.v132.REVISION and row["spec"] == spec
    assert row["arm"] == "mse-control" and row["training_device"] == "cuda" and row["trained_from_scratch"]
    assert row["training_mode"] == "fixed-profile-from-scratch"
    assert row["effective_beta"] == -2. and row["config"]["weight_reg"] == 0.
    assert row["config"]["use_context"] and not row["config"]["use_nau_actor"]
    assert row["config"]["weight_sym"] == row["config"]["jac_weight"] == row["config"]["action_anchor_weight"] == 0.
    assert row["reward_changes"] == [] and not row["policy_receives_profile_labels"]
    assert row["checkpoint_selection"] == "final-not-best" and Path(row["checkpoint"]).name == "final.pt"
    assert not row["periodic_evaluation_selects_checkpoint"] and not row["source_readout_refitted"]
    assert row["optimizer_updates_during_evaluation"] == 0
    assert row["reward_settings"] == dict(healthy_reward=1., ctrl_cost_weight=.001, forward_reward_weight=1.)
    profile_settings = study.profile_settings(profile)
    multipliers = {k: profile_settings.get(k, 1.) for k in ("body_mass", "gravity", "friction")}
    assert row["physics"]["profile"] == profile
    np.testing.assert_allclose(list(row["physics"]["actual_multipliers"].values()), list(multipliers.values()), rtol=1e-12, atol=1e-12)
    training, steps = row["training"], spec["steps"]
    assert training["interactions"] == row["final_context_count"] == steps
    assert training["optimizer_updates"] == steps-row["config"]["batch_size"]+1
    blocks = training["diagnostic_blocks"]
    assert [b["step"] for b in blocks] == list(range(spec["diagnostic_interval"], steps+1, spec["diagnostic_interval"]))
    assert sum(b["interactions"] for b in blocks) == steps
    assert sum(b["updates"] for b in blocks) == training["optimizer_updates"]
    periodic = training["periodic_evaluations"]
    assert [e["step"] for e in periodic] == [0, *range(spec["evaluation_interval"], steps, spec["evaluation_interval"])]
    for e in periodic:
        assert e["profile"] == profile and e["actual_multipliers"] == row["physics"]["actual_multipliers"]
        assert [p["seed"] for p in e["episodes"]] == list(range(spec["evaluation_seed"], spec["evaluation_seed"]+spec["evaluation_episodes"]))
    return validate_final(row, smoke, study)


def validate_final(row, smoke=False, study=worker):
    profile = row["training_profile"]
    settings = study.profile_settings(profile)
    multipliers = {k: settings.get(k, 1.) for k in ("body_mass", "gravity", "friction")}
    final, protocol = row["final_evaluation"], study.evaluation_spec(smoke)
    assert final["protocol"] == protocol and final["device"] == "cpu" and final["profile"] == profile
    assert final["optimizer_updates"] == 0 and not final["replay_loaded"]
    condition = "id" if profile == "id" else "static/"+profile
    expected = [dict(condition=condition, mode="deterministic" if noise is None else "sampled", reset_seed=reset, noise_seed=noise)
                for reset in protocol["reset_seeds"] for noise in (None, *protocol["action_noise_seeds"])]
    episodes = final["episodes"]
    assert [{k: e[k] for k in expected[0]} for e in episodes] == expected
    for e in episodes:
        assert 0 < e["length"] <= protocol["max_steps"] and e["native_reward_verified"]
        assert e["context_mode"] == "online" and e["context_updates"] == e["length"]
        assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
        assert smoke or (not e["diagnostic_cutoff"] and (e["physical_termination"] or e["time_limit_truncation"]))
        assert e["profile_counts"] == {profile: e["length"]} and e["switches"] == 0
        assert e["non_id_steps"] == (0 if profile == "id" else e["length"])
        assert len(e["profile_events"]) == 1 and e["profile_events"][0]["step"] == 1
        assert e["profile_events"][0]["profile"] == profile
        np.testing.assert_allclose(list(e["profile_events"][0]["actual_multipliers"].values()), list(multipliers.values()), rtol=1e-12, atol=1e-12)
    assert final["evaluation_interactions"] == sum(e["length"] for e in episodes)
    return episodes


def summarize(rows, study=worker):
    combinations = [(seed, profile) for seed in (3, 4) for profile in study.PROFILES]
    indexed = {(r["training_seed"], r["training_profile"]): r for r in rows}
    assert len(rows) == 6 and set(indexed) == set(combinations)
    records = []
    for seed, profile in combinations:
        row = indexed[seed, profile]
        episodes = validate(row, study=study)
        assert row["config"] == rows[0]["config"]
        assert all(row["runtime"][k] == rows[0]["runtime"][k] for k in row["runtime"] if k != "gpu")
        records.append(dict(training_seed=seed, training_profile=profile, checkpoint=row["checkpoint"],
            seconds=row["seconds"], physics=row["physics"],
            modes={mode: aggregate([e for e in episodes if e["mode"] == mode]) for mode in ("deterministic", "sampled")},
            last5k_block=row["training"]["diagnostic_blocks"][-1]))
    return dict(revision=study.REVISION, independent_training_seeds=2, final_models=6,
        training_interactions=sum(r["training"]["interactions"] for r in rows),
        optimizer_updates=sum(r["training"]["optimizer_updates"] for r in rows),
        final_evaluation_episodes=sum(len(r["final_evaluation"]["episodes"]) for r in rows), records=records,
        scope="Matched-budget fixed-profile learnability. No pooled training-seed inference, unknown-regime adaptation, physical-impossibility or baseline superiority claim.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.result_dir/f"seed{seed}_{profile}/source.json").read_text())
            for seed in (3, 4) for profile in worker.PROFILES]
    result = summarize(rows)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["records"]:
        print(row["training_seed"], row["training_profile"], row["modes"])


if __name__ == "__main__":
    main()
