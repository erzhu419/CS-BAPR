import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import summarize_general_ood_v40 as summary


class V40SummaryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((summary.STUDY / "dispatch_manifest.json").read_text())
        path = summary.DESTINATION / "formal/Hopper-v4/seed1/general_ood_v40_strength_Hopper-v4_relative-adaptive_seed1.json"
        self.candidate = json.loads(path.read_text())

    def test_real_complete_matrix_preserves_failed_gate_and_strength_comparison(self):
        result = summary.summarize(self.manifest, summary.DESTINATION)
        self.assertTrue(result["matrix_complete"])
        self.assertEqual(result["decision"], "hold-no-benefit-over-strength-matched-controls")
        self.assertEqual(result["n_training_seeds"], 1)
        for contrast in result["relative_minus_control"].values():
            self.assertLess(contrast["paired_episode_differences"]["markov_50x"]["mean"], 0)
            self.assertLess(contrast["fixed_profile_floor"], 0)
        methods = result["methods"]
        self.assertAlmostEqual(methods["fixed-minus2p3"]["diagnostics_update_weighted"]["effective_beta"],
                               -2. - .3 * (99745 - 100) / 99745)
        groups = methods["relative-adaptive"]["switch_response"]["groups"]
        self.assertEqual(groups["actual_change"]["event_count"], 17)
        self.assertEqual(groups["same_mode_redraw"]["event_count"], 2)
        self.assertFalse(methods["fixed-minus2p3"]["switch_response"]["inferred_belief"])

    def test_wrong_evaluation_protocol_and_budget_rejected(self):
        for key, value in (("training_budget", 300000), ("ood_rng_reseeded_each_episode", False)):
            changed = copy.deepcopy(self.candidate)
            changed[key] = value
            with self.assertRaises(AssertionError):
                summary.validate_run(changed, self.manifest, "relative-adaptive")

    def test_misaligned_switch_window_and_wrong_actual_change_rejected(self):
        changed = copy.deepcopy(self.candidate)
        changed["training"]["switch_diagnostics"]["events"][0]["bins"][5]["relative_start"] = 1
        with self.assertRaises(AssertionError):
            summary.validate_run(changed, self.manifest, "relative-adaptive")
        changed = copy.deepcopy(self.candidate)
        changed["training"]["switch_diagnostics"]["events"][6]["actual_change"] = True
        with self.assertRaises(AssertionError):
            summary.validate_run(changed, self.manifest, "relative-adaptive")

    def test_missing_constant_control_does_not_produce_a_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            target = destination / "formal/Hopper-v4/seed1"
            target.mkdir(parents=True)
            source = summary.DESTINATION / "formal/Hopper-v4/seed1"
            path = source / "general_ood_v40_strength_Hopper-v4_relative-adaptive_seed1.json"
            shutil.copy2(path, target / path.name)
            result = summary.summarize(self.manifest, destination)
        self.assertEqual(result["decision"], "pending-missing-cells")
        self.assertEqual(len(result["missing_cells"]), 2)

    def test_bin_and_episode_means_use_their_actual_units(self):
        bins = [{"updates": n, "means": {k: value for k in summary.SWITCH_KEYS}}
                for n, value in ((1, 2.), (3, 4.))]
        self.assertEqual(summary.bin_means(bins), {k: 3.5 for k in summary.SWITCH_KEYS})
        differences = summary.paired_differences({"rewards": [1., 4., 3.]}, {"rewards": [2., 2., 3.]})
        self.assertEqual(differences["values"], [-1., 2., 0.])
        self.assertEqual((differences["wins"], differences["ties"]), (1, 1))
        self.assertEqual(differences["mean"], 1. / 3.)


if __name__ == "__main__":
    unittest.main()
