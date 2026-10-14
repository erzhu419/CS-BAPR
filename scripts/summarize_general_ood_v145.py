"""Gravity calibration evidence, keeping own-domain and held-out peak separate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_general_ood_v145_gravity_calibration as worker
from summarize_general_ood_v129 import aggregate


def validate_final(final, profile, gravity):
    protocol = dict(worker.backend.v137.evaluation_spec(), context_mode="not-applicable")
    assert final["protocol"] == protocol and final["profile"] == profile and final["device"] == "cpu"
    assert final["optimizer_updates"] == 0 and not final["replay_loaded"]
    expected = [dict(condition="static/"+profile, mode="deterministic" if n is None else "sampled",
        reset_seed=s, noise_seed=n) for s in protocol["reset_seeds"] for n in (None, *protocol["action_noise_seeds"])]
    episodes = final["episodes"]
    assert [{k: e[k] for k in expected[0]} for e in episodes] == expected
    for e in episodes:
        assert 0 < e["length"] <= 1000 and not e["diagnostic_cutoff"]
        assert e["physical_termination"] or e["time_limit_truncation"]
        assert e["native_reward_verified"] and e["context_mode"] == "not-applicable" and e["context_updates"] == 0
        assert e["noise_scale"] == (0. if e["mode"] == "deterministic" else 1.)
        assert e["profile_counts"] == {profile: e["length"]} and e["switches"] == 0
        assert e["profile_events"] == [dict(step=1, profile=profile,
            actual_multipliers=dict(body_mass=1., gravity=gravity, friction=1.))]
    assert final["evaluation_interactions"] == sum(e["length"] for e in episodes)
    return episodes


def validate(row, id_reference):
    json.dumps(row, allow_nan=False)
    seed, profile = row["training_seed"], row["training_profile"]
    assert seed in (3, 4) and profile in worker.PROFILES
    assert row["revision"] == worker.REVISION and not row["smoke"] and row["spec"] == worker.stage_spec(profile)
    assert row["config"] == id_reference["config"] == worker.backend.SETTINGS
    assert row["reward_settings"] == id_reference["reward_settings"]
    assert row["reward_changes"] == row["healthy_termination_changes"] == []
    assert row["training_device"] == "cuda" and row["runtime"]["stable_baselines3"] == "2.7.0"
    assert row["observation_dim"] == 11 and not any(row[k] for k in ("use_context", "use_symbolic_constraints", "reward_normalization", "policy_receives_profile_labels"))
    assert row["bootstrap_mask"] == "terminated-only" and row["replay_handle_timeout_termination"]
    assert row["cumulative_training_interactions"] == 600000 and row["optimizer_updates"] == 599745
    assert 0 <= row["actual_new_training_interactions"] <= 600000
    physics, base = row["physics"], id_reference["physics"]
    assert physics["profile"] == profile and physics["actual_multipliers"] == dict(body_mass=1., gravity=worker.GRAVITY[profile], friction=1.)
    for k in ("body_mass", "geom_friction"):
        assert physics[k] == base[k]
    np.testing.assert_array_equal(physics["gravity"], np.array(base["gravity"])*worker.GRAVITY[profile])
    training = row["training"]
    assert training["step"] == 600000 and training["snapshots"] == row["snapshots"]
    assert [b["step"] for b in training["blocks"]] == list(range(5000, 600001, 5000))
    assert sum(b["interactions"] for b in training["blocks"]) == 600000
    assert sum(b["updates"] for b in training["blocks"]) == 599745
    assert [e["step"] for e in training["periodic_evaluations"]] == list(range(0, 600000, 25000))
    assert row["checkpoint_selection"] == "predeclared300k600k-not-best" and row["optimizer_updates_during_evaluation"] == 0
    assert [s["step"] for s in row["snapshots"]] == [300000, 600000]
    for s in row["snapshots"]:
        assert s["optimizer_updates"] == s["step"]-255 and Path(s["checkpoint"]).name == f"step{s['step']}.zip"
        own = s["final_evaluation"]
        validate_final(own, profile, worker.GRAVITY[profile])
        validate_final(own["heldout_peak_evaluation"], "gravity-only", row["spec"]["heldout_peak_gravity_multiplier"])


def summarize(rows, id_reference):
    assert len(rows) == 4 and {(r["training_seed"], r["training_profile"]) for r in rows} == {(s, p) for s in (3, 4) for p in worker.PROFILES}
    records = []
    for row in sorted(rows, key=lambda r: (r["training_seed"], r["training_profile"])):
        validate(row, id_reference)
        snapshots = []
        for snapshot in row["snapshots"]:
            own = snapshot["final_evaluation"]
            domains = {}
            for name, final in (("own", own), ("heldout_peak", own["heldout_peak_evaluation"])):
                episodes = final["episodes"]
                domains[name] = dict(modes={m: aggregate([e for e in episodes if e["mode"] == m]) for m in ("deterministic", "sampled")},
                    terminal_reason_sets=dict(Counter("+".join(e["final_physical_state"]["unhealthy_conditions"]) for e in episodes if e["physical_termination"])))
            snapshots.append(dict(step=snapshot["step"], checkpoint=snapshot["checkpoint"], domains=domains))
        records.append(dict(seed=row["training_seed"], profile=row["training_profile"], snapshots=snapshots,
            final_segment_interactions=row["actual_new_training_interactions"],
            restored_prefix_interactions=600000-row["actual_new_training_interactions"], final_segment_seconds=row["seconds"]))
    return dict(revision=worker.REVISION, independent_training_seeds=2,
        cumulative_training_interactions=2400000, optimizer_updates=2398980, evaluation_episodes=320,
        evaluation_interactions=sum(f["evaluation_interactions"] for r in rows for s in r["snapshots"]
            for f in (s["final_evaluation"], s["final_evaluation"]["heldout_peak_evaluation"])),
        records=records, scope="Gravity-only calibration and frozen transfer; retain resumed prefixes and seed separation. Not physical impossibility or adaptation evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v145_gravity_calibration"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads((args.result_dir/f"seed{s}_{p}/source.json").read_text()) for s in (3, 4) for p in worker.PROFILES]
    reference = json.loads((root/"results/general_ood_v144_fixed_profile_sac/seed3_id/source.json").read_text())
    result = summarize(rows, reference)
    result["formal_json_bytes"] = sum((args.result_dir/f"seed{s}_{p}/source.json").stat().st_size for s in (3, 4) for p in worker.PROFILES)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for r in result["records"]:
        print(r["seed"], r["profile"], [(s["step"], {k: d["modes"]["sampled"] for k, d in s["domains"].items()}) for s in r["snapshots"]])
