"""Opens every page of the Streamlit app in a headless session and checks what it shows.

These tests use the committed forecasts and the January dataset, so they skip if those files are absent.
"""
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "app"
needs_files = pytest.mark.skipif(
    not (ROOT / "outputs" / "forecasts" / "predictions.npz").exists()
    or not (ROOT / "data" / "processed" / "january_2024" / "demand.npy").exists(),
    reason="generated forecasts or the January dataset are missing")
pytestmark = needs_files

PAGES = ("overview", "map", "zone", "results", "reposition")
BASELINE_NAMES = ("Gradient boosting", "Ridge regression", "Persistence", "Historical average")


def open_page(name: str) -> AppTest:
    script = f"import sys\nsys.path.insert(0, {str(APP)!r})\nimport page_{name}\npage_{name}.render()\n"
    return AppTest.from_string(script, default_timeout=180).run()


def everything_shown(at: AppTest) -> str:
    parts = []
    for group in (at.markdown, at.caption, at.title, at.subheader, at.success, at.info):
        parts += [e.value for e in group]
    parts += [m.label for m in at.metric]
    for box in at.selectbox:
        parts += [str(o) for o in box.options]
    parts += [str(box.label) for box in at.selectbox]
    for frame in at.dataframe:
        parts.append(frame.value.to_string())
    return "\n".join(parts)


@pytest.mark.parametrize("name", PAGES)
def test_every_page_opens_without_an_error(name):
    at = open_page(name)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


@pytest.mark.parametrize("name", ["map", "zone", "reposition"])
def test_pages_other_than_accuracy_show_only_the_st_gnn(name):
    shown = everything_shown(open_page(name))
    for baseline in BASELINE_NAMES:
        assert baseline not in shown, f"{baseline} appears on the {name} page"


def test_the_accuracy_page_is_where_the_baselines_are_compared():
    at = open_page("results")
    shown = everything_shown(at)
    for baseline in BASELINE_NAMES:
        assert baseline in shown
    assert at.success and "lowest error at every horizon" in at.success[0].value


def test_the_map_finds_the_top_spots_and_scores_them_over_the_whole_window():
    at = open_page("map")
    labels = [m.label for m in at.metric]
    assert "Real top spots found at this time" in labels
    assert "At least one found" in labels
    assert not [b for b in at.selectbox if b.label == "Model"]            # no model chooser
    mean_hits = next(m for m in at.metric if m.label.startswith("Real top-") and "on average" in m.label)
    found, of = mean_hits.value.split(" of ")
    assert 0.0 <= float(found) <= float(of)


def test_the_zone_explorer_shows_one_forecast_and_its_error_at_every_horizon():
    at = open_page("zone")
    assert not at.multiselect
    errors = at.dataframe[0].value
    assert list(errors["Horizon"]) == ["5 min", "15 min", "30 min", "60 min"]
    assert (errors["RMSE"] > 0).all()


def test_the_repositioning_study_headline_is_a_valid_share():
    at = open_page("reposition")
    share = next(m for m in at.metric if m.label == "Share of the best possible gain")
    assert 0 <= float(share.value.rstrip("%")) <= 100
    assert any(m.label == "Empty driving" for m in at.metric)             # the cost is shown, not hidden


@pytest.mark.parametrize("policy", ["stgnn_cal", "oracle"])
def test_the_simulator_runs_for_each_policy(policy):
    at = open_page("reposition")
    assert [b.value for b in at.selectbox][0] == "stgnn_cal"
    at.selectbox[0].select(policy)
    next(s for s in at.select_slider if s.label == "Simulated duration (hours)").set_value(1)
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    moved = next(m for m in at.metric if m.label == "Vehicles repositioned")
    assert int(moved.value.replace(",", "")) > 0
    unserved = [m for m in at.metric if m.label == "Requests left unserved"][-1]
    assert 0 <= float(unserved.value.rstrip("%")) <= 100
