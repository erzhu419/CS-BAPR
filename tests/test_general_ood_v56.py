from types import SimpleNamespace

import numpy as np

from csbapr.belief.departure import DepartureReadout, NormalAgeReference
from scripts import probe_general_ood_v55_learning_calibration as v55
from scripts import probe_general_ood_v56_reset_calibration as v56


def test_episode_ordinals_change_after_done_not_on_terminal_transition():
    np.testing.assert_array_equal(v56.episode_ordinals([False, True, False, False, True]), [0, 0, 1, 1, 1])


def test_population_windows_separate_initial_and_reset_samples():
    dones = np.zeros(1200, dtype=bool)
    dones[999] = True
    ages = np.r_[np.arange(1000), np.arange(200)]
    signals = np.r_[np.full(1000, .8), np.full(200, .02)]
    reference = SimpleNamespace(age_reference=NormalAgeReference.fit_normal(signals, ages))
    records = v56.population_windows([(signals, ages, v56.episode_ordinals(dones))], reference)
    for r in records[:5]:
        assert r['pooled_threshold'] == .8
        assert r['initial'] == dict(count=17, q95=.8, maximum=.8)
        assert r['after_reset'] == dict(count=16 if r['age'] == 192 else 17, q95=.02, maximum=.02)
        assert r['threshold_exceeds_after_reset_max']
    assert records[-1]['after_reset']['count'] == 0
    assert records[-1]['threshold_exceeds_after_reset_max'] is None


def test_window_details_report_suppressed_signal_and_actual_reset():
    dones = np.zeros(1200, dtype=bool)
    dones[999] = True
    ages = np.r_[np.arange(1000), np.arange(200)]
    signals = np.full(1200, .12)
    reference = SimpleNamespace(age_reference=NormalAgeReference([0, 999], [.8, .8], [1, 1], 2))
    windows = v56.window_details(signals, ages, dones, reference, 800)
    last = windows['last64']
    assert (last['start'], last['end'], last['age_min'], last['age_max']) == (1136, 1200, 136, 199)
    assert last['episode_ordinal_min'] == last['episode_ordinal_max'] == 1
    assert last['mean_signal'] > 0 and last['min_threshold'] > last['max_signal']
    assert last['above_threshold_fraction'] == last['mean_coefficient'] == 0
    assert windows['reset1_first50']['start'] == 1000
    assert windows['reset1_before64']['end'] == 1000
    departure = DepartureReadout(reference.age_reference)
    expected = [departure.update(s, a) for s, a in zip(signals, ages)]
    assert windows['full']['mean_coefficient'] == np.mean(expected)


def test_replay_budget_and_seed_roles_are_original_formal_cells():
    cell = dict(split='test', seed=98723, updating=True, mode='high_grav',
                change_minus_nochange={'learning_calibrated': dict(first_post50=.08, last64=0.)})
    original = dict(stage='formal', stage_spec=v55.stage_spec('formal'), records=[cell])
    cells = v56.replay_cells(original)
    assert [(c['split'], c['seed'], c['mode']) for c in cells] == [
        ('calibration', 98700, 'normal'), ('calibration', 98701, 'normal'), ('test', 98723, 'high_grav')]
    assert all(c['updating'] for c in cells)
    assert len(cells) * original['stage_spec']['steps'] == 3600
