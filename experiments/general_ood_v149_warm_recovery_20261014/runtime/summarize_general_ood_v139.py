"""Pool equal-sized diagnostic strata, not whole-replay or training averages."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_general_ood_v139_multinode import CASES


def summarize(root):
    rows, indices = [], None
    for case in CASES:
        path = root/case/"diagnostics.json"
        row = json.loads(path.read_text())
        diagnostic = row["diagnostic"]
        assert row["revision"] == "general_ood_v139_frozen_diagnostics" and row["case"] == case
        assert not row["smoke"] and row["device"] == "cpu"
        assert diagnostic["training_interactions"] == diagnostic["optimizer_updates"] == diagnostic["environment_steps"] == 0
        assert diagnostic["source_optimizer_updates"] == 299745 and diagnostic["replay_size"] == 300000
        bins = diagnostic["bins"]
        actual = [b["indices"] for b in bins]
        assert len(bins) == 5 and all(len(x) == 128 for x in actual)
        assert indices is None or actual == indices
        indices = actual
        pooled = {}
        for name in bins[0]["quantities"]:
            values = [b["quantities"][name] for b in bins]
            assert all(v["nonfinite"] == 0 for v in values)
            count = sum(v["count"] for v in values)
            pooled[name] = {key: sum(v[key]*v["count"] for v in values)/count for key in ("mean", "abs_mean")}
        actor = {}
        for location in ("actor", "next_actor"):
            values = [b[location] for b in bins]
            actor[location] = dict(saturation_fraction=sum(v["saturation_fraction"] for v in values)/5.,
                mean_density_gap=sum(v["stable_minus_native"]["mean"] for v in values)/5.,
                max_abs_density_gap=max(max(abs(v["stable_minus_native"][key]) for key in ("min", "max")) for v in values))
        rows.append(dict(case=case, source=row["source"], result=str(path.resolve()),
            alpha=diagnostic["alpha"], quantities=pooled, actor=actor,
            policy_q_std_mean=(pooled["policy_q"]["mean"]-pooled["policy_q_mean_minus_2std"]["mean"])/2.,
            terminated_count=sum(b["terminated_count"] for b in bins), samples=640,
            seconds=row["seconds"], json_bytes=path.stat().st_size))
    return dict(revision="general_ood_v139_frozen_diagnostics", models=rows,
        training_interactions=0, optimizer_updates=0, environment_steps=0,
        formal_replay_samples=3840, formal_json_bytes=sum(r["json_bytes"] for r in rows),
        averaging="Equal samples from unequal-age intervals; not a whole-replay or historical-training mean",
        limitation="Final frozen model diagnostics, not historical causal attribution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("results/general_ood_v139_frozen_diagnostics"))
    args = parser.parse_args()
    result = summarize(args.result_root)
    (args.result_root/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
