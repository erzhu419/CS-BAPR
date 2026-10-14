import gymnasium as gym
import numpy as np

from csbapr.envs.extreme_ood import ExtremeOODConfig, ExtremeOODWrapper


def wrapper():
    return ExtremeOODWrapper(
        gym.make("Pendulum-v1"),
        profiles=[{"name": name} for name in ("id", "a", "b", "c")],
        config=ExtremeOODConfig(seed=80000, schedule="markov",
                                switch_every_steps=1, switch_probability=1.),
    )


def trace(env, steps):
    return [env.step(np.zeros(1))[4]["ood_profile"] for _ in range(steps)]


def test_seeded_episode_is_independent_of_previous_episode_length():
    first, second = wrapper(), wrapper()
    try:
        first.reset(seed=80000)
        second.reset(seed=80000)
        trace(first, 7)
        trace(second, 19)
        first.reset(seed=80001)
        second.reset(seed=80001)
        assert trace(first, 30) == trace(second, 30)
    finally:
        first.close()
        second.close()


def test_repeated_seed_repeats_profile_stream_and_new_seed_changes_it():
    env = wrapper()
    try:
        env.reset(seed=80000)
        expected = trace(env, 30)
        env.reset(seed=80000)
        assert trace(env, 30) == expected
        env.reset(seed=80001)
        assert trace(env, 30) != expected
    finally:
        env.close()
