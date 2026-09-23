"""Szenario: geometrische Basis wortgleich zur Hill-Climbing-Demo (hc_scenario.generate), erweitert um Bedarfe und
Kapazität. Auswertungstests folgen, sobald vrpn_evaluation.py existiert (siehe Reihenfolge im Bau-Plan)."""

import numpy as np
import pytest

import vrpn_constants as C
import vrpn_scenario as S


def test_instance_shape_depot_and_area():
    inst = S.generate(60, 0, 3)
    assert inst.xy.shape == (61, 2) and inst.n == 60 and inst.n_nodes == 61
    assert inst.xy[0].tolist() == [50.0, 50.0]
    assert inst.xy.min() >= 0.0 and inst.xy.max() <= C.AREA
    assert inst.demands.shape == (61,) and inst.demands[0] == 0


def test_instance_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(40, 25, 5), S.generate(40, 25, 5), S.generate(40, 25, 6)
    assert np.array_equal(a.xy, b.xy) and np.array_equal(a.demands, b.demands)
    assert not np.array_equal(a.xy, c.xy)


def test_geometry_matches_the_hill_climbing_root_exactly():
    """Die geometrische Basis (Lage von Depot und Stopps) muss bytegleich zu hc_scenario.generate sein - die
    Bedarfe/Kapazität sind eine reine ERGÄNZUNG, keine Änderung der zugrunde liegenden Geometrie. Überspringt sich
    selbst ohne den Geschwisterordner (eigenständiges Repo, CI/frische Klone haben ihn nicht daneben liegen)."""
    import sys
    from pathlib import Path
    hc_dir = Path(__file__).resolve().parent.parent.parent / "hill-climbing-demo"
    if not (hc_dir / "hc_scenario.py").exists():
        pytest.skip("hill-climbing-demo liegt nicht als Geschwisterordner vor")
    sys.path.insert(0, str(hc_dir))
    import hc_scenario as HC
    for n, ballung, seed in [(60, 0, 35), (40, 25, 5), (10, 100, 0)]:
        ours = S.generate(n, ballung, seed)
        theirs = HC.generate(n, ballung, seed)
        assert np.array_equal(ours.xy, theirs.xy)


def test_grouped_stops_lie_closer_together_than_uniform_ones():
    def mean_nn(share):
        vals = []
        for seed in range(10):
            xy = S.generate(80, share, seed).xy[1:]
            d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            vals.append(d.min(axis=1).mean())
        return float(np.mean(vals))
    assert mean_nn(100) < 0.7 * mean_nn(0)


def test_demands_are_within_bounds_and_depot_has_none():
    inst = S.generate(100, 0, 7)
    assert inst.demands[0] == 0
    assert inst.demands[1:].min() >= C.DEMAND_MIN and inst.demands[1:].max() <= C.DEMAND_MAX


def test_total_demand_property():
    inst = S.generate(50, 0, 1)
    assert inst.total_demand == pytest.approx(float(inst.demands.sum())) and inst.total_demand > 0


def test_capacity_is_stored_as_given():
    inst = S.generate(30, 0, 2, capacity=123.0)
    assert inst.capacity == 123.0
