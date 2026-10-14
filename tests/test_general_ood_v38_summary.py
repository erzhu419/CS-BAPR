import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import summarize_general_ood_v38 as summary


class V38SummaryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((summary.STUDY / "dispatch_manifest.json").read_text())
        path = summary.DESTINATION / "formal/Pendulum-v1/seed0/general_ood_v38_belief_Pendulum-v1_corrected-belief_seed0.json"
        self.data = json.loads(path.read_text())

    def test_real_matrix_retains_negative_seed_and_exposure(self):
        result = summary.summarize(self.manifest, summary.DESTINATION)
        self.assertEqual(result["completed_runs"], 12)
        self.assertEqual(result["decision"], "hold-mechanism-diagnosis")
        hopper = result["methods"]["Hopper-v4"][summary.CANDIDATE]
        self.assertLess(hopper["returns"]["id"]["seed_values"][1], 100)
        self.assertEqual(hopper["diagnostics"][1]["ood_non_id_transition_count"], 0)

    def test_budget_context_and_config_mismatches_rejected(self):
        for key in ("training_budget", "loaded_training_context_count", "ood_switch_steps"):
            changed = copy.deepcopy(self.data)
            changed[key] += 1
            with self.assertRaises(AssertionError):
                summary.validate_run(changed, self.manifest, "Pendulum-v1", summary.CANDIDATE, 0)
        changed = copy.deepcopy(self.data)
        changed["config"]["penalty_scale"] = 0
        with self.assertRaises(AssertionError):
            summary.validate_run(changed, self.manifest, "Pendulum-v1", summary.CANDIDATE, 0)

    def test_positive_ood_does_not_override_floor_regression(self):
        control = {k: {"seed_values": values} for k, values in (
            ("id", [2, 3]), ("markov_50x", [1, 4]), ("fixed_profile_floor", [-1, -2]))}
        gate = summary.decision_gate({"legacy-belief": control, "fixed-penalty": control})
        self.assertTrue(gate["ood_wins_both_seeds_against_both_controls"])
        self.assertFalse(gate["admit_further_confirmation"])
        control["fixed_profile_floor"]["seed_values"] = [0, 1]
        self.assertTrue(summary.decision_gate({"control": control})["admit_further_confirmation"])
        control["id"]["seed_values"] = [-1, -2]
        self.assertFalse(summary.decision_gate({"control": control})["admit_further_confirmation"])

    def test_missing_results_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            result = summary.summarize(self.manifest, Path(directory))
        self.assertEqual(len(result["missing_cells"]), 12)
        self.assertEqual(result["decision"], "pending-missing-cells")
        self.assertEqual(result["eligible_environments"], [])


if __name__ == "__main__":
    unittest.main()
