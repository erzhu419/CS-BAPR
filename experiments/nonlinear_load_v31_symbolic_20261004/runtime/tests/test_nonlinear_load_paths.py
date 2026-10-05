import json
from unittest.mock import patch

import pytest
import torch

from scripts.evaluate_nonlinear_load_v12 import load_actor_parameters, reproduce


def test_inference_json_restores_float32_actor_parameters_exactly():
    actor = torch.nn.Linear(6, 2)
    expected = {k: v.clone() for k, v in actor.state_dict().items()}
    payload = json.loads(json.dumps({k: v.tolist() for k, v in expected.items()}))
    restored = torch.nn.Linear(6, 2)
    load_actor_parameters(restored, payload)
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert not restored.training


def test_reproduction_keeps_both_endpoint_checks_and_rejects_a_changed_policy():
    actual = {a: {"returns": [-float(a)]} for a in (1, 50)}
    expected = {"actor": {"1": -1.0, "50": -50.0}}
    with patch("scripts.evaluate_nonlinear_load_v12.evaluate_ood", return_value=actual) as evaluate, \
            patch("scripts.evaluate_nonlinear_load_v12.evaluate_controls", return_value={}):
        assert reproduce(None, expected)["passed"]
        assert evaluate.call_args.kwargs["seed_base"] == 50000
        expected["actor"]["50"] = -51.0
        with pytest.raises(AssertionError):
            reproduce(None, expected)
