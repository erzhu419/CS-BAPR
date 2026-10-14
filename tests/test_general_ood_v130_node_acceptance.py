from copy import deepcopy
import json

import pytest

from scripts import accept_general_ood_v130_nodes as acceptance


def setup_results(root):
    runtime = {"python": "3.10.14", "torch": "2.5.1+cu121"}
    episode = {key: 1 for key in acceptance.EXACT + acceptance.SCALARS}
    episode["final_physical_state"] = {"unhealthy_conditions": []}
    evaluation = dict(training_seed=2, runtime=runtime, spec={"smoke": True},
        optimizer_updates=0, new_training_interactions=0, evaluations=[deepcopy(episode) for _ in range(6)])
    rows = {"smoke_eval_seed2/replication.json": evaluation}
    for node in acceptance.NODES:
        rows[f"setup_{node}/environment.json"] = dict(node=node, environment_ready=True, runtime=runtime)
        rows[f"smoke_{node}_continuation_seed2/mean_retention.json"] = dict(training_seed=2,
            smoke=True, arm="unchanged-control", actual_new_training_interactions=512,
            actual_new_optimizer_updates=512, training={"interactions": 300512},
            config={"device": "cuda"}, runtime={"gpu": "RTX 3080 Ti"},
            mean_retention_coefficient=0., effective_beta=-2.)
        rows[f"smoke_{node}_evaluation_seed2/replication.json"] = evaluation
    for path, row in rows.items():
        destination = root/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(row))


def test_release_requires_both_nodes_and_keeps_preferences_empty(tmp_path):
    setup_results(tmp_path)
    result = acceptance.validate(tmp_path)
    assert len(result["nodes"]) == 2
    assert result["allowed_nodes"] == ["node007", "jtl110gpu", "jtl110gpu2"]
    assert result["preferred_node"] is result["preferred_gpu"] is None


def test_release_rejects_native_evaluation_difference(tmp_path):
    setup_results(tmp_path)
    path = tmp_path/"smoke_jtl110gpu2_evaluation_seed2/replication.json"
    row = json.loads(path.read_text())
    row["evaluations"][0]["episode_return"] += .01
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError, match="episode_return"):
        acceptance.validate(tmp_path)
