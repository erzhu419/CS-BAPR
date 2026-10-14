"""Isolate existing mega mass, gravity and friction at the v137 budget."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import run_general_ood_v137_fixed_profile as v137

v132 = v137.v132
REVISION = "general_ood_v138_physics_components"
FACTORS = {"mass-only": "body_mass", "gravity-only": "gravity", "friction-only": "friction"}
PROFILES = tuple(FACTORS)
evaluation_spec = v137.evaluation_spec


def profile_settings(profile):
    assert profile in PROFILES
    mega = v137.profile_settings("mega_event")
    result = dict(name=profile, latent_load=50., obs_scale=1., action_scale=1.,
                  body_mass=1., gravity=1., friction=1.)
    factor = FACTORS[profile]
    result[factor] = mega[factor]
    return result


def stage_spec(profile, smoke=False):
    settings = profile_settings(profile)
    return dict(v137.stage_spec("mega_event", smoke), study_revision=REVISION, training_profile=profile,
        profile_family="structured-component", source_profile_family="structured", source_profile="mega_event",
        active_factor=FACTORS[profile], physics_factors={k: settings[k] for k in FACTORS.values()})


def make_wrapped_env(condition):
    profile = condition.removeprefix("static/")
    settings = profile_settings(profile)
    return v137.v135.ExtremeOODWrapper(gym.make("Hopper-v4"), profiles=[settings],
        config=v137.v135.ExtremeOODConfig(intensity=50., profile_family="structured-component",
            schedule="static", static_profile=profile, include_sensor_noise=False,
            include_action_noise=False, include_reward_noise=False))


def make_env(profile):
    wrapper = make_wrapped_env("static/"+profile)
    wrapper.reset(seed=0)
    metadata = dict(profile=profile, actual_multipliers=v137.v135.actual_multipliers(wrapper),
        body_mass=wrapper.unwrapped.model.body_mass.tolist(), gravity=wrapper.unwrapped.model.opt.gravity.tolist(),
        geom_friction=wrapper.unwrapped.model.geom_friction.tolist())
    return wrapper.env, metadata


def final_evaluation(agent, profile, smoke=False):
    return v137.final_evaluation(agent, profile, smoke, env_factory=make_wrapped_env)


def run(args):
    v137.run(args, revision=REVISION, env_factory=make_env, spec_factory=stage_spec,
             final_evaluator=final_evaluation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
