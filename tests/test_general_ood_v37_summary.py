import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v37_summary", ROOT / "scripts/summarize_general_ood_v37.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class V37SummaryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "environments": ["Pendulum-v1", "Hopper-v4"],
            "methods": [module.CANDIDATE, "csbapr-no-sindy-stable", "bapr-stable-context", "resac"],
            "training_seeds": [0, 1, 2], "evaluation_seed_start": 70000,
            "evaluation_episodes": 20, "ood_intensity": 50.0,
            "ood_profile_family": "structured", "ood_schedule": "markov",
            "ood_switch_probability": 1.0,
            "ood_switch_steps": {"Pendulum-v1": 50, "Hopper-v4": 200},
        }
        def block(count):
            return {"mean": 1.0, "std": 0.0, "rewards": [1.0] * count, "lengths": [200] * count}
        self.data = {
            "revision": module.REVISION, "environment": "Pendulum-v1",
            "method": module.CANDIDATE, "training_seed": 0,
            "checkpoint": f"{module.CANDIDATE}_Pendulum-v1_0_final.pt",
            "evaluation_only": True, "optimizer_updates_during_evaluation": 0,
            **{k: v for k, v in self.manifest.items() if k not in ("environments", "methods", "training_seeds", "ood_switch_steps")},
            "ood_switch_steps": 50,
            "results": {"id": block(20), "markov_50x": block(20)},
            "fixed_profile_results": {p: block(5) for p in module.PROFILES["Pendulum-v1"]},
        }

    def test_valid_schema_and_protocol_mismatch(self):
        module.validate_run(self.data, self.manifest, "Pendulum-v1", module.CANDIDATE, 0)
        changed = copy.deepcopy(self.data)
        changed["ood_switch_steps"] = 100
        with self.assertRaises(AssertionError):
            module.validate_run(changed, self.manifest, "Pendulum-v1", module.CANDIDATE, 0)

    def test_truncated_or_nonfinite_evaluation_rejected(self):
        for values in ([1.0], [float("nan")] * 20):
            changed = copy.deepcopy(self.data)
            changed["results"]["id"]["rewards"] = values
            with self.assertRaises(AssertionError):
                module.validate_run(changed, self.manifest, "Pendulum-v1", module.CANDIDATE, 0)

    def test_hopper_profiles_are_environment_specific(self):
        data = copy.deepcopy(self.data)
        data.update(environment="Hopper-v4", ood_switch_steps=200,
                    checkpoint=f"{module.CANDIDATE}_Hopper-v4_0_final.pt")
        block = next(iter(data["fixed_profile_results"].values()))
        data["fixed_profile_results"] = {p: copy.deepcopy(block) for p in ("easy_surface", "heavy_traffic", "mega_event", "slippery_rush")}
        module.validate_run(data, self.manifest, "Hopper-v4", module.CANDIDATE, 0)

    def test_training_seed_interval_not_episode_pooled(self):
        stats = module.seed_statistics([1.0, 2.0, 3.0])
        self.assertEqual(stats["n_training_seeds"], 3)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["ci95"][1], 2.0 + 4.302652729911275 / 3**0.5)
        self.assertIsNone(module.seed_statistics([1.0])["ci95"])

    def test_missing_matrix_never_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = module.summarize(self.manifest, Path(directory))
        self.assertEqual(len(summary["missing_cells"]), 24)
        self.assertFalse(summary["matrix_complete"])
        self.assertEqual(summary["decision"], "pending-missing-cells")
        self.assertTrue(all(x is None for x in summary["all_seed_baseline_superiority_by_environment"].values()))


if __name__ == "__main__":
    unittest.main()
