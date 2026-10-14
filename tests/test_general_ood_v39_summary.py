import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import summarize_general_ood_v39 as summary


class V39SummaryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((summary.STUDY / "dispatch_manifest.json").read_text())

    def test_real_repaired_matrix_retains_failed_control_and_gate(self):
        result = summary.summarize(self.manifest, summary.DESTINATION)
        self.assertEqual(result["n_training_seeds"], 1)
        self.assertEqual(result["decision"], "hold-no-benefit-over-fixed-penalty")
        self.assertTrue(result["raw_vs_fixed_minus3"]["paired_evaluation_equal"])
        self.assertTrue(all(x == 0 for x in result["raw_vs_fixed_minus3"]["max_absolute_training_trace_differences"].values()))
        self.assertEqual(result["methods"]["raw-adaptive"]["ood_non_id_transition_count"], 0)
        self.assertLess(result["relative_minus_control"]["fixed-minus2"]["markov_50x"], 0)
        self.assertTrue(result["id_and_fixed_profile_evaluation_preserved"])

    def test_wrong_budget_and_unrepaired_evaluation_rejected(self):
        path = summary.DESTINATION / "paired_eval/Hopper-v4/seed1/general_ood_v39_surprise_Hopper-v4_relative-adaptive_seed1_paired_eval.json"
        data = json.loads(path.read_text())
        for key, value in (("training_budget", 300000), ("ood_rng_reseeded_on_episode_reset", False)):
            changed = copy.deepcopy(data)
            changed[key] = value
            with self.assertRaises(AssertionError):
                summary.validate_run(changed, self.manifest, "relative-adaptive", paired=True)

    def test_original_only_matrix_waits_for_repaired_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            shutil.copytree(summary.DESTINATION / "formal", destination / "formal")
            result = summary.summarize(self.manifest, destination)
        self.assertEqual(len(result["missing_cells"]), 4)
        self.assertFalse(result["matrix_complete"])
        self.assertEqual(result["decision"], "pending-missing-cells")

    def test_diagnostic_weighting_uses_update_counts(self):
        blocks = [{"step": 1000, "updates": 1, "surprise": 2., "recent_completed_episode_return": None},
                  {"step": 2000, "updates": 3, "surprise": 4., "recent_completed_episode_return": 10.}]
        self.assertEqual(summary.weighted_metrics(blocks), {"surprise": 3.5})


if __name__ == "__main__":
    unittest.main()
