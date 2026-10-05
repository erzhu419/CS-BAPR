from types import SimpleNamespace

import numpy as np
import pytest
import torch

from csbapr.sindy.load_feedback import known_load_reference
from scripts import test_lqr_convergence as runner


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_direct_feedback_uses_paired_paths_and_normalized_actions(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    reference = known_load_reference().to(device)

    def action(observation, deterministic):
        assert deterministic
        with torch.no_grad():
            return reference(torch.as_tensor(observation, dtype=torch.float32, device=device)).cpu().numpy()

    expected = runner.evaluate_ood(SimpleNamespace(select_action=action), [1, 50],
                                   n_eval=2, seed_base=20000, environment="random_load")
    actual = runner.evaluate_feedback(reference, [1, 50], n_eval=2, seed_base=20000)
    assert actual["amplitude_results"] == expected
    np.testing.assert_array_equal(actual["gain"], reference.gain.cpu().numpy())
    assert actual["amplitude_results"][50]["evaluation_seeds"] == [20000, 20001]
    other = runner.evaluate_feedback(reference, [50], n_eval=2, seed_base=20002)
    assert actual["amplitude_results"][50]["returns"] != other["amplitude_results"][50]["returns"]


def test_wrong_evaluation_paths_are_rejected_on_resume(tmp_path):
    from csbapr.agent import CSBAPRAgent

    config = runner.make_config("csbapr-centered-mlp-no-target-shift", "cpu", "random_load")
    agent = CSBAPRAgent(6, 2, config)
    env = runner.make_env("random_load", 1, 3)
    protocol = {"evaluation_seed_start": 20000}
    path = str(tmp_path / "last.pt")
    runner.save_training_checkpoint(agent, env, path, next_ep=0, env_steps=0, history=[], protocol=protocol)
    with pytest.raises(ValueError, match="exact protocol"):
        runner.restore_training_checkpoint(agent, env, path, {"evaluation_seed_start": 10000})
