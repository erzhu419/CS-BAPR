import copy
import json

import numpy as np
import pytest
import torch

from scripts import calibrate_nonlinear_load_v27 as calibration
from tests.test_nonlinear_load_direction_v26 import payloads_for


def test_id_only_signed_labels_and_paired_actions(monkeypatch, tmp_path):
    payloads, _ = payloads_for(19)
    actor, critic, gain, terminal = calibration.load_networks(payloads["early"], "cpu")
    spec = calibration.calibration_spec(True)
    initial = np.array([[.1, 0, 0, 0, .02, 0], [0, -.2, 0, 0, 0, -.03]], dtype=np.float32)

    def rollout(actors, critics, gains, terminals, radius, load, paths, noises, branches):
        assert load == 1 and paths == [61000, 61001] and noises == [93000, 93001]
        assert branches == {"plus": .005, "minus": -.005}
        return initial, {"early": {"continuations": {
            branch: {"soft_q_return": np.ones((2, 2)) + shift * np.array([.5, -.75]),
                     "soft_actor_return": np.ones((2, 2)) + shift * np.array([8., 9.])}
            for branch, shift in branches.items()}}}

    monkeypatch.setattr(calibration, "stochastic_rollout", rollout)
    data, metadata = calibration.collect_calibration(actor, critic, gain, terminal, spec, tmp_path)
    np.testing.assert_allclose(data["targets"].numpy(), [.5, -.75])
    assert metadata["positive_direction_labels"] == metadata["negative_direction_labels"] == 1
    assert metadata["controller_episodes"] == 8 and metadata["environment_interactions"] == 1600
    assert data["plus_actions"].shape == (2, 2, 2)
    states = torch.from_numpy(initial)
    direction = calibration.mean_direction(actor, gain, states)[2]
    for draw, seed in enumerate(spec["noise_seeds"]):
        noise = torch.randn((2, 2), generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            plus, _ = calibration.policy_sample(actor, states, noise, .005 * direction)
            minus, _ = calibration.policy_sample(actor, states, noise, -.005 * direction)
        torch.testing.assert_close(data["plus_actions"][draw], plus, rtol=0, atol=0)
        torch.testing.assert_close(data["minus_actions"][draw], minus, rtol=0, atol=0)
    raw = json.loads((tmp_path / "radius1.0_load1.json").read_text())
    assert raw["load"] == 1 and set(raw["path_seeds"]).isdisjoint(range(60000, 60100))


def test_direction_noise_average_and_individual_value_anchor():
    spec = calibration.calibration_spec(True)
    prediction = torch.zeros(3, 2, 2, 2)
    prediction[:, 0, 0] = 2 * spec["h"] * torch.tensor([1., 3.])
    prediction[:, 0, 1] = 2 * spec["h"] * torch.tensor([3., 1.])
    prediction.requires_grad_(True)
    original = torch.zeros_like(prediction)
    total, direction, anchor = calibration.fit_losses(prediction, original, torch.tensor([1., 2.]), spec)
    assert float(direction.detach()) == pytest.approx(.5)
    assert float(anchor.detach()) == pytest.approx(float(prediction.detach().square().mean()))
    assert float(total.detach()) == pytest.approx(float(direction.detach() + anchor.detach()))
    total.backward()
    assert torch.isfinite(prediction.grad).all()


def test_checkpoint_resume_preserves_exact_next_updates_and_frozen_control(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    original = calibration.EnsembleQNet(6, 2, 8, 3).requires_grad_(False)
    data = {"states": torch.randn(4, 6), "plus_actions": torch.randn(2, 4, 2) * .1,
            "minus_actions": torch.randn(2, 4, 2) * .1, "targets": torch.tensor([.2, -.3, .4, -.1])}
    spec = {**calibration.calibration_spec(True), "updates": 10}
    protocol = {"calibration_spec": spec, "training_seed": 19}
    snapshot = copy.deepcopy(original.state_dict())
    full, report, _ = calibration.fit_critic(original, data, {}, protocol, tmp_path / "full.pt")
    split = tmp_path / "split.pt"
    calibration.fit_critic(original, data, {}, protocol, split, stop_after=5)
    resumed, resumed_report, _ = calibration.fit_critic(original, data, {}, protocol, split, resume=split)
    for a, b in zip(full.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.grad is None and b.grad is None
    assert report["updates"] == resumed_report["updates"] == 10
    assert report["history"][-1] == resumed_report["history"][-1]
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)


@pytest.mark.parametrize("seed", calibration.TRAINING_SEEDS)
def test_source_actor_and_terminal_scale_identity(seed):
    payloads, old_manifest = payloads_for(seed)
    payload = payloads["early"]
    source = {"revision": calibration.SOURCE_REVISION, "training_seed": seed,
        "checkpoint_selection": "last", "source_ids": {"early": f"task{seed}"},
        "config": old_manifest["config"], "environment_settings": {},
        "parameters_unchanged": True, "optimizer_updates": 0, "fitting_interactions": 0,
        "preflight_only": False, "gamma": .99, "finite_difference_steps": list(calibration.STEPS),
        "terminal": {"early": {k: payload[k] for k in ("alpha", "reward_std", "weighted_lambda")}}}
    source["terminal"]["early"]["beta"] = -2 - 2 * payload["weighted_lambda"]
    manifest = {"source_revision": calibration.SOURCE_REVISION, "source_tasks": {str(seed): {"training_id": f"task{seed}"}},
                "config": old_manifest["config"], "environment_settings": {}}
    calibration.validate_source(payload, source, manifest, seed)
    source["terminal"]["early"]["reward_std"] += 1
    with pytest.raises(AssertionError):
        calibration.validate_source(payload, source, manifest, seed)
