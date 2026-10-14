from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_general_ood_v130_multinode as worker


def test_native_cuda_paths_precede_scheduler_jax_paths(tmp_path, monkeypatch):
    library = tmp_path/"lib/python3.10/site-packages/nvidia/cublas/lib"
    library.mkdir(parents=True)
    monkeypatch.setattr(worker.sys, "prefix", str(tmp_path))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old-jax/lib:/driver/lib")
    assert worker.native_environment()["LD_LIBRARY_PATH"] == f"{library}:/old-jax/lib:/driver/lib"
    assert worker.os.environ["LD_LIBRARY_PATH"] == "/old-jax/lib:/driver/lib"


def test_path_mapping_changes_paths_not_protocol_or_scalar_values():
    original = dict(checkpoint=str(worker.CENTRAL_RUNTIME/"run_checkpoints/seed3/final.pt"),
        source_checkpoint=str(worker.CENTRAL_RUNTIME/"prefix/run_checkpoints/seed3/latest.pt"),
        config={"weight_reg": 0., "device": "cuda"}, spec={"mean_retention_coefficient": 1},
        training={"interactions": 300000}, evaluations=[{"episode_return": -5.5}])
    local = worker.rewrite_paths(deepcopy(original), worker.CENTRAL_PROJECT, worker.LOCAL_PROJECT)
    assert local["checkpoint"] == str(worker.LOCAL_PROJECT/worker.RUNTIME_REL/"run_checkpoints/seed3/final.pt")
    assert worker.rewrite_paths(local, worker.LOCAL_PROJECT, worker.CENTRAL_PROJECT) == original


def test_stage_only_pulls_full_state_for_training(tmp_path, monkeypatch):
    path = worker.CENTRAL_RUNTIME/"run_results/seed3/source.json"
    checkpoint = worker.CENTRAL_RUNTIME/"run_checkpoints/seed3/final.pt"
    data = dict(training_seed=3, checkpoint=str(checkpoint), config={"weight_reg": 0.})
    copies = []
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(data)))
    monkeypatch.setattr(worker, "copy_artifact", lambda p: copies.append(Path(p)))
    monkeypatch.setattr(worker, "local_path", lambda p: tmp_path/Path(p).name)
    worker.stage_source(path)
    assert copies == [checkpoint]
    copies.clear()
    worker.stage_source(path, full=True)
    assert copies == [checkpoint, worker.CENTRAL_RUNTIME/"run_checkpoints/seed3/latest.pt"]
    restored = json.loads((tmp_path/"source.json").read_text())
    assert restored["config"] == data["config"]
    assert restored["checkpoint"] == str(worker.LOCAL_PROJECT/worker.RUNTIME_REL/"run_checkpoints/seed3/final.pt")


def test_publish_final_first_keeps_replay_on_compute_node(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "LOCAL_PROJECT", tmp_path)
    result = tmp_path/"runtime/run_results/seed3_unchanged-control"
    checkpoint = tmp_path/"runtime/run_checkpoints/seed3_unchanged-control"
    result.mkdir(parents=True)
    path = result/"mean_retention.json"
    row = dict(checkpoint=str(checkpoint/"final.pt"),
        reference_checkpoint=str(tmp_path/"runtime/run_checkpoints/seed3/final.pt"),
        training={"interactions": 350000}, mean_retention_coefficient=0.)
    path.write_text(json.dumps(row))
    calls = []
    monkeypatch.setattr(worker.subprocess, "run", lambda command, **kwargs: calls.append(command))
    worker.publish(SimpleNamespace(stage="continuation", result_dir=result, checkpoint_dir=checkpoint))
    transfers = [c for c in calls if c[0] == "rsync"]
    assert len(transfers) == 2
    assert str(checkpoint/"final.pt") in transfers[0] and str(path) in transfers[1]
    assert not any(str(checkpoint/"latest.pt") in c for c in calls)
    final = json.loads(path.read_text())
    assert final["training"] == row["training"]
    assert final["checkpoint"] == str(worker.CENTRAL_PROJECT/"runtime/run_checkpoints/seed3_unchanged-control/final.pt")
    assert final["resumable_checkpoint"] == str(checkpoint/"latest.pt")
