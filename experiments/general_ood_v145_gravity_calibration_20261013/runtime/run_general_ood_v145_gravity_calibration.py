"""Gravity-only learning calibration, retaining the original held-out peak."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = (HERE.parents[0]/"experiments/general_ood_v145_gravity_calibration_20261013/runtime"
           if HERE.name == "scripts" else HERE)
sys.path.insert(0, str(RUNTIME))

import run_general_ood_v144_fixed_profile_sac as backend

REVISION = "general_ood_v145_gravity_calibration"
GRAVITY = {"gravity-1p5": 1.5, "gravity-2p0": 2.}
PROFILES = tuple(GRAVITY)


def profile_settings(profile):
    settings = backend.v138.profile_settings("gravity-only")
    return dict(settings, name=profile, gravity=GRAVITY[profile])


def make_wrapped_env(profile):
    return backend.v137.v135.ExtremeOODWrapper(backend.gym.make("Hopper-v4"),
        profiles=[profile_settings(profile)], config=backend.v137.v135.ExtremeOODConfig(
            intensity=50., profile_family="structured-component", schedule="static", static_profile=profile,
            include_sensor_noise=False, include_action_noise=False, include_reward_noise=False))


def make_env(profile):
    wrapper = make_wrapped_env(profile)
    wrapper.reset(seed=0)
    raw = wrapper.unwrapped
    metadata = dict(profile=profile, actual_multipliers=backend.v137.v135.actual_multipliers(wrapper),
        body_mass=raw.model.body_mass.tolist(), gravity=raw.model.opt.gravity.tolist(),
        geom_friction=raw.model.geom_friction.tolist())
    return wrapper.env, metadata


def stage_spec(profile, smoke=False):
    return dict(backend.stage_spec("gravity-only", smoke), revision=REVISION, profile=profile,
        gravity_multiplier=GRAVITY[profile], heldout_peak_profile="gravity-only",
        heldout_peak_gravity_multiplier=backend.v138.profile_settings("gravity-only")["gravity"],
        training_intervention="gravity-multiplier-only", reward_changes=[], healthy_termination_changes=[])


def final_evaluation(checkpoint, profile, smoke, training_device):
    own = backend.final_evaluation(checkpoint, profile, smoke, training_device, env_factory=make_wrapped_env)
    peak = backend.final_evaluation(checkpoint, "gravity-only", smoke, training_device)
    return dict(**own, heldout_peak_evaluation=peak)


def run(args):
    backend.run(args, revision=REVISION, spec_factory=stage_spec, env_factory=make_env,
                evaluation_env_factory=make_wrapped_env, final_evaluator=final_evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
