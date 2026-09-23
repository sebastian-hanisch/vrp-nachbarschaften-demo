"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, Randwerte, Würfel-Knopf, Permalink-Grenzen,
Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import vrpn_constants as C
import vrpn_evaluation as ev
import vrpn_interroute as IR

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(vrpn_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if vrpn_step != 1:
        at.select_slider(key="vrpn_step").set_value(vrpn_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_measured_default():
    at = _run()
    _ok(at)
    assert _metric(at, "Verbesserung ggü. Konstruktion") == "0.26 %"
    assert _metric(at, "Routen") == "5"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["capacity_slider"] == p["capacity"] and at.session_state["budget_select"] == p["budget"] and at.session_state["n_slider"] == p["n"]
    assert list(at.session_state["moves_select"]) == p["active_moves"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
def test_every_step_runs(step):
    at = _run(n_slider=20, budget_select=25000, vrpn_step=step)
    _ok(at)
    assert at.get("plotly_chart") and at.session_state["vrpn_step"] == step


def test_deselecting_all_moves_falls_back_to_intra_with_a_warning():
    at = _run(moves_select=[])
    _ok(at)
    assert any("Intra-Route wird automatisch" in w.value for w in at.warning)


def test_dice_button_changes_the_seed():
    at = _run(budget_select=10000)
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


@pytest.mark.parametrize("kw", [dict(n_slider=200, budget_select=50000), dict(n_slider=10, ballung_slider=100, budget_select=10000),
                                 dict(capacity_slider=C.CAPACITY_MIN, budget_select=25000), dict(capacity_slider=C.CAPACITY_MAX, budget_select=25000),
                                 dict(moves_select=["intra"], budget_select=25000), dict(moves_select=["relocate"], budget_select=25000)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["ballung"] = "40"
    at.query_params["capacity"] = "9999999"
    at.query_params["budget"] = "12345"
    at.query_params["moves"] = "intra,swap"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["ballung_slider"] == 50
    assert at.session_state["capacity_slider"] == C.CAPACITY_MAX and at.session_state["budget_select"] == C.DEFAULT_BUDGET
    assert at.session_state["moves_select"] == ("intra", "swap")


def test_sweeps_run_on_demand():
    at = _run(n_slider=10, budget_select=10000)
    at.selectbox(key="sweep_select").set_value("capacity").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_experiments_run_on_demand(monkeypatch):
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    monkeypatch.setattr(ev, "SCALING_POLICIES", (("Budget 4 Tausend", lambda n: 4000), ("Budget 300 · Stopps", lambda n: 300 * n)))
    at = _run(n_slider=10, budget_select=10000)
    for key, flag in (("ablation_start", "ablation_on"), ("scaling_start", "scaling_on")):
        next(b for b in at.button if b.key == key).click().run()
        _ok(at)
        assert at.session_state[flag]


def test_footer_and_grenzen_are_present():
    at = _run(budget_select=10000)
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Kapazität liegt in einem mittleren Bereich" in m.value for m in at.markdown)
