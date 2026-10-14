from scripts import summarize_general_ood_v146 as merger


def test_recovery_is_not_full_survival_and_paired_deltas_use_all_episodes():
    episodes = [dict(length=n, reset_seed=i, noise_seed=None, episode_return=float(n), mean_velocity=1.,
        physical_termination=n < 1000, time_limit_truncation=n == 1000,
        recovery=dict(exposed=True, recovery_evaluable=True, recovered=True)) for i, n in enumerate((250, 1000))]
    baseline = {(e["reset_seed"], None): dict(length=1000) for e in episodes}
    metrics = merger.group_metrics(episodes, baseline)
    assert metrics["recovered"] == 2 and metrics["delayed_falls_after_recovery"] == 1
    assert metrics["time_limits"] == 1 and metrics["mean_paired_length_delta"] == -375.


def test_pre_exposure_failure_is_retained_in_paired_mean():
    e = dict(length=80, reset_seed=1, noise_seed=None, episode_return=80., mean_velocity=1.,
        physical_termination=True, time_limit_truncation=False,
        recovery=dict(exposed=False, recovery_evaluable=False, recovered=False))
    metrics = merger.group_metrics([e], {(1, None): dict(length=80)})
    assert metrics["episodes"] == metrics["falls"] == 1 and metrics["exposed"] == 0
    assert metrics["mean_paired_length_delta"] == 0.
