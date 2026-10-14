import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import summarize_general_ood_v41 as summary


class V41SummaryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((summary.STUDY / "dispatch_manifest.json").read_text())
        path = summary.DESTINATION / "preflight/general_ood_v41_frozen_signal_fixed-minus2.json"
        self.preflight = json.loads(path.read_text())

    def test_paired_statistics_are_not_training_seed_statistics(self):
        result = summary.contrast_statistics([1., 3., 4.], [2., 2., 4.])
        self.assertEqual(result["pairs"], 3)
        self.assertEqual(result["positive_pairs"], 1)
        self.assertEqual(result["median_paired_difference"], 0.)
        self.assertEqual(result["paired_difference_range"], [-1., 1.])

    def test_action_coupling_or_false_pairing_rejected(self):
        modified = copy.deepcopy(self.preflight)
        modified["auxiliary_belief_controls_actions"] = True
        with self.assertRaises(AssertionError):
            summary.validate_run(modified, self.manifest, "preflight")
        modified = copy.deepcopy(self.preflight)
        modified["records"][1]["change_minus_nochange"]["step-overlap"]["pre256"]["surprise"] = 1.
        with self.assertRaises(AssertionError):
            summary.validate_run(modified, self.manifest, "preflight")

    def test_final_time_limit_endings_are_not_called_physical_failures(self):
        normal = [{"episode_endings_by_256": [0, 0, 0, 0, 0, 1]}]
        earlier_reset = [{"episode_endings_by_256": [0, 0, 1, 0, 0, 1]}]
        result = summary.ending_statistics(earlier_reset, normal, 5)
        self.assertTrue(result["actual_time_limit_impossible"])
        self.assertFalse(result["nochange_time_limit_impossible"])
        self.assertTrue(summary.ending_statistics(normal, normal, 4)["actual_time_limit_impossible"])

    def test_incomplete_policy_matrix_stays_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "preflight").mkdir()
            source = summary.DESTINATION / "preflight/general_ood_v41_frozen_signal_fixed-minus2.json"
            shutil.copy2(source, destination / "preflight" / source.name)
            result = summary.summarize(self.manifest, destination, "preflight")
        self.assertEqual(result["decision"], "pending-missing-policies")
        self.assertEqual(result["missing_policies"], ["relative-adaptive"])


if __name__ == "__main__":
    unittest.main()
