from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import analyze_general_ood_v105_behavior_support as worker
from scripts import analyze_general_ood_v79_linear_support as support
from test_general_ood_v68 import trace
from test_general_ood_v78 import banks


def test_population_includes_failures_normal_frozen_and_historical_controls():
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root/"results/general_ood_v98_regime_replay/replay_diverse_recipe/regime_replay.json").read_text())["records"]
    population = worker.select_windows(records)
    assert len(population) == 62
    assert sum(r["group"] == "v95_development_frozen" for r, _ in population) == 30
    assert sum(r["mode"] == "normal" for r, _ in population) == 13
    assert len(worker.select_windows(records, True)) == 4
    with pytest.raises(AssertionError):
        worker.select_windows(records[:-1])


def test_extending_mean_preserves_incumbent_and_forward_and_has_no_target_leakage(banks):
    incumbent = banks["updating"]
    old_data = incumbent.tree.data.copy()
    old_residuals = {m: v.copy() for m, v in incumbent.residuals.items()}
    payloads = {b: {m: trace(135230, m, 80) for m in worker.v98.v94.MODES} for b in worker.v104.BEHAVIORS}
    old_groups, new_groups = np.full(incumbent.tree.n, 1), np.full(80, 135230)
    means, groups = worker.extend_means(incumbent, payloads, old_groups, new_groups)
    np.testing.assert_array_equal(incumbent.tree.data, old_data)
    for m in worker.v98.v94.MODES:
        np.testing.assert_array_equal(incumbent.residuals[m], old_residuals[m])
    query = trace(15, "compound", 8)
    before, predictions = worker.diagnose(query, incumbent, means["exploratory"], groups["exploratory"],
                                          "compound", incumbent.tree.n, support, True)
    changed = deepcopy(query); changed["next_states"] += 123
    after, changed_predictions = worker.diagnose(changed, incumbent, means["exploratory"], groups["exploratory"],
                                                "compound", incumbent.tree.n, support, True)
    for m in worker.v98.v94.MODES:
        np.testing.assert_array_equal(predictions[m], changed_predictions[m])
    assert before["prediction_error_rms"] != after["prediction_error_rms"]
    assert 0 <= before["new_reference_neighbor_fraction"] <= 1


def test_native_payload_contract_rejects_unpaired_inputs(tmp_path):
    for seed in worker.v104.SEEDS:
        directory = tmp_path/f"run_checkpoints/seed{seed}"
        directory.mkdir(parents=True)
        data = {b: {m: {k: trace(seed, m, 2200)[k] for k in worker.v104.KEYS}
                    for m in worker.v98.v94.MODES} for b in worker.v104.BEHAVIORS}
        torch.save(dict(revision=worker.v104.REVISION, seed=seed, spec=worker.v104.SPEC,
            extensions=data, groups={b: np.full(2200, seed) for b in worker.v104.BEHAVIORS},
            sources={b: [] for b in worker.v104.BEHAVIORS}), directory/"reference_inputs.pt")
        scalar = tmp_path/f"run_results/seed{seed}/reference_collection.json"
        scalar.parent.mkdir(parents=True)
        scalar.write_text(json.dumps(dict(revision=worker.v104.REVISION, seed=seed, spec=worker.v104.SPEC,
            passed=True, smoke=False, paired_prefixes=True, equal_behavior_budgets=True,
            exact_true_mode_replays_verified=True, actual_actions_enter_replay_and_context=True,
            evaluation_paths_used_for_reference=False, counterfactuals_added_to_RL_replay=False,
            per_behavior_reference_inputs={b: 2200 for b in worker.v104.BEHAVIORS},
            per_behavior_budgets={b: dict(actual_transitions=12000) for b in worker.v104.BEHAVIORS})))
    payload, groups, _ = worker.load_extensions(tmp_path)
    assert len(groups) == len(payload["standard"]["normal"]["states"]) == 4400
    path = tmp_path/f"run_checkpoints/seed{worker.v104.SEEDS[0]}/reference_inputs.pt"
    saved = torch.load(path, weights_only=False)
    saved["extensions"]["exploratory"]["heavy"]["executed_actions"][0, 0] += 1
    torch.save(saved, path)
    with pytest.raises(AssertionError):
        worker.load_extensions(tmp_path)


def test_summary_primary_comparison_is_equal_budget_behavior_contrast():
    records = []
    for seed, mode, window in worker.FAILURES:
        diagnostics = {a: dict(prediction_error_rms=v, support=dict(nearest_radius=dict(mean=2.),
            radius64=dict(mean=3.)), new_reference_neighbor_fraction=0.2)
            for a, v in zip(worker.ARMS, (1., .8, .6))}
        records.append(dict(group="v95_development_updating", seed=seed, mode=mode, updating=True,
                            window=window, old_coverage_failure=True, diagnostics=diagnostics))
    result = worker.summarize(records)["six_old_failures"]
    assert result["windows"] == 6
    primary = result["comparisons"]["exploratory_vs_standard"]
    assert primary["prediction_rms_change_percent"] == pytest.approx(-25)
    assert primary["improved_windows"] == 6
