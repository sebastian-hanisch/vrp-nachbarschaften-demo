"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-
Instanzen belegt, mit denselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`/
`ev.compare_active_moves`/`ev.scaling_table`) - NIE über ein Ad-hoc-Skript mit abweichender Zufalls-Bindung (die
Lehre aus der lin-kernighan-demo dieser Linie). Positive UND negative Aussagen: Inter-Route-Züge helfen klar
(positiv) - aber NICHT monoton mit der Kapazität, wie die Vorab-Vermutung nahelegte (negativ, ehrlicher
Kernbefund: das Optimum liegt bei MITTLERER Kapazität, nicht bei kleiner oder sehr großer)."""

from functools import lru_cache

import pytest

import vrpn_constants as C
import vrpn_evaluation as ev
import vrpn_interroute as IR


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Ablation bei Kapazität 120 (dort trägt Inter-Route am meisten bei) -------------------------------------------------------------------------


@pytest.mark.parametrize("moves,expected,tol", [
    ((), 0.0, 0.05), (("intra",), 0.867, 0.5), (("intra", "relocate"), 1.024, 0.5), (("intra", "swap"), 0.992, 0.5),
    (("intra", "2opt_star"), 2.357, 0.7), (("intra", "cross"), 1.455, 0.6), (IR.MOVE_TYPES, 2.472, 0.7),
])
def test_ablation_numbers_at_capacity_120(moves, expected, tol):
    row = cfg(capacity=120, active_moves=moves)
    near(row["improvement"], expected, tol)


def test_two_opt_star_alone_captures_most_of_the_full_neighborhood_value():
    full = cfg(capacity=120, active_moves=IR.MOVE_TYPES)["improvement"]
    star_only = cfg(capacity=120, active_moves=("intra", "2opt_star"))["improvement"]
    relocate_only = cfg(capacity=120, active_moves=("intra", "relocate"))["improvement"]
    swap_only = cfg(capacity=120, active_moves=("intra", "swap"))["improvement"]
    assert star_only > 0.8 * full                                                # 2-opt* allein erreicht > 80% des vollen Werts
    assert star_only > relocate_only + 0.5 and star_only > swap_only + 0.5       # deutlich vor Relocate/Swap allein


def test_cross_exchange_alone_stays_weaker_than_two_opt_star_alone_despite_generalizing_it():
    """Ehrlicher, erklärbarer Befund: CROSS-exchange verallgemeinert 2-opt* BEWEISBAR (siehe test_interroute.py),
    ist hier aber auf Segmentlänge MAX_SEGMENT=3 begrenzt - 2-opt* kann beliebig lange Routen-Enden tauschen, das
    begrenzte CROSS-exchange nicht, und bleibt deshalb ALLEIN schwächer."""
    star_only = cfg(capacity=120, active_moves=("intra", "2opt_star"))["improvement"]
    cross_only = cfg(capacity=120, active_moves=("intra", "cross"))["improvement"]
    assert star_only > cross_only + 0.3


# --- Kapazitäts-Sweep: NICHT monoton, Optimum bei mittlerer Kapazität (zentraler ehrlicher Befund) -----------------------------------------------


@pytest.mark.parametrize("capacity,expected,tol", [(15, 0.363, 0.3), (30, 1.232, 0.5), (60, 0.976, 0.5), (120, 2.472, 0.7), (250, 3.683, 0.9), (600, 1.164, 0.5)])
def test_capacity_sweep_numbers(capacity, expected, tol):
    row = cfg(capacity=capacity)
    near(row["improvement"], expected, tol)


def test_capacity_value_is_not_monotone_the_optimum_is_in_the_middle():
    vals = {cap: cfg(capacity=cap)["improvement"] for cap in (15, 30, 60, 120, 250, 600)}
    best_cap = max(vals, key=vals.get)
    assert best_cap not in (15, 600)                                             # das Optimum liegt NICHT an einem der beiden Ränder
    assert vals[best_cap] > vals[15] + 0.5 and vals[best_cap] > vals[600] + 0.5


def test_very_large_capacity_gives_zero_inter_route_benefit():
    """Bei einer Route gibt es keine andere Route mehr zum Tauschen - Intra-Route allein reicht dann aus, der
    volle Nachbarschafts-Zusatz ist strukturell null."""
    intra_only = cfg(capacity=600, active_moves=("intra",))["improvement"]
    full = cfg(capacity=600, active_moves=IR.MOVE_TYPES)["improvement"]
    assert full == pytest.approx(intra_only, abs=0.05)


# --- Budget-Sweep: konvergiert deutlich unter 100 Tausend Bewertungen ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget,expected,tol", [(10000, 0.487, 0.3), (25000, 0.841, 0.3), (50000, 0.907, 0.3), (100000, 0.976, 0.2), (200000, 0.976, 0.2)])
def test_budget_sweep_numbers(budget, expected, tol):
    row = cfg(budget=budget)
    near(row["improvement"], expected, tol)


def test_budget_plateaus_by_one_hundred_thousand():
    at_100k = cfg(budget=100000)["improvement"]
    at_2m = cfg(budget=2000000)["improvement"]
    assert at_100k == pytest.approx(at_2m, abs=0.05)


# --- Skalierung ------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,expected,tol", [(20, 1.186, 0.5), (40, 2.11, 0.6), (60, 0.976, 0.5), (100, 0.981, 0.5), (150, 0.569, 0.4), (200, 0.197, 0.3)])
def test_scaling_numbers_at_fixed_two_hundred_thousand_budget(n, expected, tol):
    row = ev.run_config(ev.Settings(), n=n, budget=200000)
    near(row["improvement"], expected, tol)


def test_improvement_declines_at_large_n_with_fixed_capacity_and_budget():
    small = ev.run_config(ev.Settings(), n=60, budget=200000)["improvement"]
    large = ev.run_config(ev.Settings(), n=200, budget=200000)["improvement"]
    assert large < small - 0.3


# --- Sonstiges --------------------------------------------------------------------------------------------------------------------------------


def test_preset_count_matches_the_readme():
    assert len(C.PRESETS) == 6


def test_savings_construction_gives_a_positive_finite_cost():
    inst, D = ev.instance(60, 0, C.DEFAULT_SEED, C.DEFAULT_CAPACITY)
    a = ev.analyse(ev.Settings())
    assert 0 < a.construction_cost < D.sum()
