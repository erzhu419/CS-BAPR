"""Slow actor updates with the original critic rate and unchanged physics."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_general_ood_v140_critic_rate as v140

v137, v138, v132 = v140.v137, v140.v138, v140.v132
REVISION = "general_ood_v141_actor_rate"
PROFILES = v140.PROFILES
ACTOR_LR = 3e-5
profile_settings = v140.profile_settings
make_env = v140.make_env
evaluation_spec = v140.evaluation_spec
final_evaluation = v140.final_evaluation


def make_agent(env, device, arm):
    agent = v132.make_agent(env, device, arm)
    assert agent.config.lr_actor == 1e-4 and agent.config.lr_critic == 3e-4
    agent.config.lr_actor = ACTOR_LR
    for group in agent.actor_optimizer.param_groups:
        group["lr"] = ACTOR_LR
    return agent


def stage_spec(profile, smoke=False):
    assert profile in PROFILES
    baseline = v137 if profile == "id" else v138
    return dict(baseline.stage_spec(profile, smoke), study_revision=REVISION,
                lr_actor=ACTOR_LR, control_lr_actor=1e-4, control_revision=baseline.REVISION)


def run(args):
    v137.run(args, revision=REVISION, env_factory=make_env, spec_factory=stage_spec,
             final_evaluator=final_evaluation, agent_factory=make_agent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4), required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
