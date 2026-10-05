from copy import deepcopy

import numpy as np
import pytest

from scripts.evaluate_nonlinear_load_v13 import check_reproduction, make_radius_env


def test_radius_changes_only_initial_physical_state_and_preserves_load_paths():
    radii = (0.2, 0.5, 1.0)
    envs = [make_radius_env(50, 60000, radius) for radius in radii]
    observations = [env.reset()[0] for env in envs]
    for radius, env, observation in zip(radii, envs, observations):
        assert np.linalg.norm(env.state) <= radius
        np.testing.assert_allclose(env.state / radius, envs[0].state / radii[0])
        np.testing.assert_array_equal(observation[4:], observations[0][4:])
    actions = (np.zeros(2), np.array([0.3, -0.2]), np.array([-0.7, 0.9]))
    for _ in range(200):
        transitions = [env.step(action) for env, action in zip(envs, actions)]
        for transition in transitions[1:]:
            np.testing.assert_array_equal(transition[4]["applied_disturbance"], transitions[0][4]["applied_disturbance"])
            assert transition[4]["storm"] == transitions[0][4]["storm"]


def test_original_radius_reproduction_checks_every_paired_return():
    seeds = [60000, 60001]
    values = {str(a): {"evaluation_seeds": seeds, "returns": [-float(a), -2.0 * a]} for a in (1, 50)}
    source = {"amplitude_results": values, "feedback_controls": {"known_inverse": {"amplitude_results": values}}}
    assert check_reproduction(source, source, seeds)["passed"]
    changed = deepcopy(source)
    changed["feedback_controls"]["known_inverse"]["amplitude_results"]["50"]["returns"][1] -= 1.0
    with pytest.raises(AssertionError):
        check_reproduction(changed, source, seeds)
