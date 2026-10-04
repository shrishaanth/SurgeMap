from argparse import Namespace

import pytest

from run_evaluation import METRICS, build_tasks, frontier_gain, summarize


def make_run(policy, seed, wait, fleet=1000, theta=0.1, unmet=0.5, empty=100.0):
    run = {"policy": policy, "fleet": fleet, "seed": seed, "theta": theta}
    run.update({m: 0.0 for m in METRICS})
    run.update(mean_wait_penalized=wait, unmet_rate=unmet, empty_minutes=empty)
    return run


def test_summary_averages_over_seeds_and_pairs_with_persistence():
    runs = [make_run("persistence", 0, 10.0), make_run("persistence", 1, 12.0),
            make_run("stgnn", 0, 9.0), make_run("stgnn", 1, 10.0)]
    rows = {r["policy"]: r for r in summarize(runs)}
    assert rows["stgnn"]["mean_wait_penalized"] == pytest.approx(9.5)
    assert rows["stgnn"]["mean_wait_penalized_std"] == pytest.approx(0.7071, abs=1e-3)
    # paired: (9 - 10) and (10 - 12) -> mean -1.5
    assert rows["stgnn"]["mean_wait_penalized_vs_persistence"] == pytest.approx(-1.5)
    assert "mean_wait_penalized_vs_persistence" not in rows["persistence"]


def test_summary_keeps_fleet_sizes_and_thetas_apart():
    runs = [make_run("stgnn", 0, 5.0, fleet=1000), make_run("stgnn", 0, 7.0, fleet=2000),
            make_run("stgnn", 0, 6.0, fleet=1000, theta=0.5)]
    assert len(summarize(runs)) == 3


def test_frontier_gain_compares_policies_at_equal_driving_cost():
    # persistence: wait = 10 - 0.01 * cost.  stgnn drives less for the same wait, i.e. is
    # 1 minute better at every cost.  oracle is 2 minutes better.
    runs = []
    for i, cost in enumerate((100.0, 200.0, 300.0, 400.0)):
        runs.append(make_run("persistence", 0, 10.0 - 0.01 * cost, theta=0.1 * (i + 1), empty=cost))
        runs.append(make_run("stgnn", 0, 9.0 - 0.01 * cost, theta=0.1 * (i + 1), empty=cost))
        runs.append(make_run("oracle", 0, 8.0 - 0.01 * cost, theta=0.1 * (i + 1), empty=cost))
    gain = frontier_gain(summarize(runs), fleet=1000)
    assert gain["stgnn"]["gain"] == pytest.approx(1.0)
    assert gain["oracle"]["gain"] == pytest.approx(2.0)
    assert gain["stgnn"]["share_of_oracle_gain"] == pytest.approx(0.5)
    assert "persistence" not in gain


def test_frontier_gain_ignores_a_policy_with_too_few_points_or_no_overlap():
    runs = [make_run("persistence", 0, 10.0 - i, theta=0.1 * i, empty=100.0 * i) for i in (1, 2, 3)]
    few = runs + [make_run("stgnn", 0, 5.0, theta=0.1, empty=100.0)]
    assert frontier_gain(summarize(few), fleet=1000) == {}
    far = runs + [make_run("stgnn", 0, 5.0, theta=0.2 * i, empty=1000.0 + i) for i in (1, 2, 3)]
    assert "stgnn" not in frontier_gain(summarize(far), fleet=1000)


def test_task_grid_has_no_duplicates_and_includes_baseline_and_sweep():
    args = Namespace(fleet_sizes=[1000, 2000], seeds=[0, 1], policies=["persistence", "stgnn"],
                     theta=0.1, thetas=[0.05, 0.1, 0.5], sweep_fleet=2000,
                     sweep_policies=["persistence", "stgnn"])
    tasks = build_tasks(args)
    keys = [(t["policy"], t["fleet"], t["seed"], t["theta"]) for t in tasks]
    assert len(keys) == len(set(keys))
    assert sum(t["policy"] == "none" for t in tasks) == 4
    assert ("stgnn", 2000, 1, 0.5) in keys and ("stgnn", 1000, 1, 0.5) not in keys
    assert ("persistence", 2000, 0, 0.1) in keys
