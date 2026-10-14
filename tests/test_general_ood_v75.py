import argparse
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
import torch

from scripts import analyze_general_ood_v75_support as worker
from scripts.probe_general_ood_v42 import DynamicsResidual


def bank():
    features = DynamicsResidual(2, True)
    features.input_mean = np.arange(14, dtype=float)
    features.input_scale = np.full(14, 2.)
    physical = np.concatenate((np.zeros((64, 14)), np.ones((64, 14))))
    tree = cKDTree((physical-features.input_mean)/features.input_scale)
    return SimpleNamespace(features=features, tree=tree, density_transitions=64)


def query(value=0., steps=2):
    return dict(states=np.full((steps, 11), value), executed_actions=np.full((steps, 3), value),
                next_states=np.zeros((steps, 11)))


def test_neighbors_use_fixed_scaling_and_input_only():
    b = bank()
    q = query()
    original = worker.support(q, b, 64)
    assert original['mean_radius64'] == 0 and original['added_neighbor_fraction'] == 0
    extended = worker.support(query(1.), b, 64)
    assert extended['mean_radius64'] == 0 and extended['added_neighbor_fraction'] == 1
    q['next_states'] += 100
    assert worker.support(q, b, 64) == original
    q['executed_actions'][:] = .5
    separated = worker.support(q, b, 64)
    assert separated['mean_neighbor_state_rms'] == 0
    assert separated['mean_neighbor_action_rms'] == .25
    np.testing.assert_allclose(separated['mean_radius64'], np.sqrt(3)*.25)


def test_clipping_uses_only_hopper_velocity_coordinates():
    states = np.zeros((4, 11))
    states[0, 0] = 10
    states[1, 5] = 10
    states[2, 10] = -10
    states[3, 6] = 9.999
    assert worker.clipped_fraction(states) == .5


def test_scalar_worker_keeps_frozen_updating_groups(tmp_path, monkeypatch):
    spec = dict(worker.v74.SPEC, steps=100, switch_step=20)
    monkeypatch.setattr(worker.v74, 'SPEC', spec)
    checkpoint = tmp_path/'run_checkpoints/preflight'
    checkpoint.mkdir(parents=True)
    references = {c: dict(window16=SimpleNamespace(bank=bank())) for c in worker.v74.CONDITIONS}
    original = references['v72']['window16'].bank
    original.tree = cKDTree(original.tree.data[:64])
    for refs in references.values():
        refs['recursive'] = SimpleNamespace(bank=refs['window16'].bank)
    torch.save(dict(revision=worker.v74.REVISION, spec=spec, references=references), checkpoint/'density_coverage_references.pt')
    cells = [dict(group='v62_development_replay', seed=134020, mode='normal', updating=u) for u in (False, True)]
    scores = [dict(**cell, observers={c: dict(recursive={w: dict(mode_accuracy=.9)
        for w in ('first_post50', 'last64')}) for c in worker.v74.CONDITIONS}) for cell in cells]
    directory = tmp_path/'run_results/screen'
    directory.mkdir(parents=True)
    (directory/'density_coverage.json').write_text(json.dumps(dict(revision=worker.v74.REVISION, spec=spec, records=scores)))
    monkeypatch.setattr(worker.v74, 'evaluation_inputs', lambda args, stored: [(cell, i) for i, cell in enumerate(cells)])
    traces = [query(steps=100), query(steps=100)]
    traces[1]['states'][20:, 5] = 10
    monkeypatch.setattr(worker.v74.v68, 'load_trace', lambda path: deepcopy(traces[path]))
    worker.run(argparse.Namespace(v74_dir=tmp_path, result_dir=tmp_path/'result'))
    output = json.loads((tmp_path/'result/support_attribution.json').read_text())
    assert len(output['records']) == 2 and output['environment_steps'] == output['optimizer_updates'] == 0
    assert not output['reference_refitted'] and not output['temporal_rules_changed']
    assert output['reference_clipping']['v72']['added_clipped_velocity_fraction'] is None
    for record in output['records']:
        assert record['windows']['first_post50']['clipped_input_velocity_fraction'] == float(record['updating'])
        assert record['windows']['last64']['conditions']['slippery_state']['recursive_mode_accuracy'] == .9
    assert not any(k in json.dumps(output) for k in ('"states"', '"executed_actions"', '"next_states"', '"coefficients"'))
