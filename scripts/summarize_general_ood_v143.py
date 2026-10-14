"""Frozen terminal-window evidence with matched v142 episode references."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_general_ood_v143_terminal_trace as worker
import summarize_general_ood_v137 as previous

SCALAR_METRICS = ("episode_return", "mean_velocity", "mean_policy_std", "mean_action_noise_l2", "mean_context_l2")
PHYSICAL_METRICS = ("height", "angle", "max_abs_qvel")


def validate(row, source):
    json.dumps(row, allow_nan=False)
    assert row["revision"] == worker.REVISION and not row["smoke"]
    seed, origin, target = worker.CASES[worker.CASE_NAMES.index(row["case"])]
    assert (source["training_seed"], source["training_profile"]) == (seed, origin)
    assert source["revision"] == worker.source_worker.REVISION and not source["smoke"]
    assert row["source"] == dict(revision=source["revision"], checkpoint=source["checkpoint"],
        seed=seed, profile=origin, config=source["config"], training_interactions=600000, optimizer_updates=599745)
    assert all(row["runtime"][k] == source["runtime"][k] for k in row["runtime"])
    assert row["evaluation_device"] == "cpu" and not row["cuda_initialized"] and not row["replay_loaded"]
    assert row["new_training_interactions"] == row["optimizer_updates"] == 0
    assert row["reward_changes"] == row["healthy_termination_changes"] == row["physics_profile_changes"] == []
    protocol = worker.v137.evaluation_spec()
    assert row["spec"] == dict(protocol, case=row["case"], source_profile=origin, evaluation_profile=target, smoke=False)
    final = dict(protocol=protocol, profile=target, device="cpu", optimizer_updates=0,
        replay_loaded=False, episodes=row["episodes"], evaluation_interactions=row["evaluation_interactions"])
    episodes = previous.validate_final(dict(training_profile=target, final_evaluation=final), study=worker.source_worker.source_worker)
    for index, episode in enumerate(episodes):
        trace = episode["terminal_trace"]
        assert trace["steps"] == episode["length"] and trace["sampled_step_stride"] == 4 and trace["max_snapshots"] == 17
        expected = [i for i in range(1, episode["length"]+1) if i % 4 == 0 or i == episode["length"]][-17:]
        assert [p["step"] for p in trace["tail"]] == expected
        assert trace["healthy_height_bounds"] == [.7, None] and trace["healthy_angle_bounds"] == [-.2, .2]
        assert trace["actuator_joint_names"] == ["thigh_joint", "leg_joint", "foot_joint"]
        assert 0 <= trace["max_abs_normalized_action"] <= 1.
        assert len(trace["action_saturation_fraction"]) == 3 and all(0 <= x <= 1 for x in trace["action_saturation_fraction"])
        last = trace["tail"][-1]
        assert last["physical_termination"] == episode["physical_termination"] and last["time_limit_truncation"] == episode["time_limit_truncation"]
        np.testing.assert_allclose(last["after"]["qpos"][1:3],
            [episode["final_physical_state"]["height"], episode["final_physical_state"]["angle"]], rtol=1e-10, atol=1e-10)
        if origin == target:
            baseline = source["final_evaluation"]["episodes"][index]
            observed = {k: v for k, v in episode.items() if k != "terminal_trace"}
            assert set(observed) == set(baseline)
            for key, value in observed.items():
                if key in SCALAR_METRICS:
                    continue
                elif key == "final_physical_state":
                    assert value["unhealthy_conditions"] == baseline[key]["unhealthy_conditions"]
                else:
                    assert value == baseline[key], key
    return episodes


def summarize(rows, sources):
    assert len(rows) == 8 and {r["case"] for r in rows} == set(worker.CASE_NAMES)
    records = []
    for row in sorted(rows, key=lambda r: worker.CASE_NAMES.index(r["case"])):
        source = sources[row["source"]["seed"], row["source"]["profile"]]
        episodes = validate(row, source)
        modes = {}
        for mode in ("deterministic", "sampled"):
            group = [e for e in episodes if e["mode"] == mode]
            modes[mode] = dict(**previous.aggregate(group),
                mean_abs_normalized_action=np.mean([e["terminal_trace"]["mean_abs_normalized_action"] for e in group], axis=0).tolist(),
                action_saturation_fraction=np.mean([e["terminal_trace"]["action_saturation_fraction"] for e in group], axis=0).tolist(),
                mean_tail_abs_joint_torque=np.mean([np.mean(np.abs([p["joint_torque"] for p in e["terminal_trace"]["tail"]]), axis=0) for e in group], axis=0).tolist(),
                mean_tail_start_joint_qpos=np.mean([e["terminal_trace"]["tail"][0]["before"]["qpos"][3:] for e in group], axis=0).tolist(),
                mean_tail_final_joint_qpos=np.mean([e["terminal_trace"]["tail"][-1]["after"]["qpos"][3:] for e in group], axis=0).tolist(),
                terminal_reason_sets=dict(Counter("+".join(e["final_physical_state"]["unhealthy_conditions"]) for e in group if e["physical_termination"])))
        reference_deltas = None
        if row["spec"]["source_profile"] == row["spec"]["evaluation_profile"]:
            pairs = list(zip(episodes, source["final_evaluation"]["episodes"]))
            reference_deltas = dict(
                scalar={k: max(abs(a[k]-b[k]) for a, b in pairs) for k in SCALAR_METRICS if k in episodes[0]},
                physical={k: max(abs(a["final_physical_state"][k]-b["final_physical_state"][k]) for a, b in pairs) for k in PHYSICAL_METRICS})
        records.append(dict(case=row["case"], source=row["source"], seconds=row["seconds"], modes=modes,
            reference_max_abs_deltas=reference_deltas))
    return dict(revision=worker.REVISION, evaluation_episodes=160,
        evaluation_interactions=sum(r["evaluation_interactions"] for r in rows),
        matched_reference_episodes=120, matched_reference_lengths_terminations=120,
        reference_comparison="Exact protocol/length/termination/reason matches; floating trajectory deltas retained, not required to be bit-identical across native CPU nodes",
        new_training_interactions=0, optimizer_updates=0, records=records,
        scope="Two original training seeds; reset/noise repeats are not new training seeds. One ID4 cross-physics positive control; not adaptation, physical impossibility or algorithm superiority.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/general_ood_v143_terminal_trace"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads((args.result_dir/case/"terminal_trace.json").read_text()) for case in worker.CASE_NAMES]
    sources = {(s, p): json.loads((root/f"results/general_ood_v142_budget_extension/seed{s}_{p}/source.json").read_text())
               for s in (3, 4) for p in worker.source_worker.PROFILES}
    result = summarize(rows, sources)
    result["formal_json_bytes"] = sum((args.result_dir/case/"terminal_trace.json").stat().st_size for case in worker.CASE_NAMES)
    (args.result_dir/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for r in result["records"]:
        print(r["case"], {m: dict(length=v["mean_length"], falls=v["falls"], saturation=v["action_saturation_fraction"]) for m, v in r["modes"].items()})
