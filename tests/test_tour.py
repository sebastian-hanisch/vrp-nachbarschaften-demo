"""Routendarstellung, Kosten, Machbarkeit und das eigenständig hergeleitete Pfad-2-opt (Depot an beiden Enden
fest) - gegen Brute-Force geprüft, dazu die ehrlich gemessene Einschränkung im TSP-Sonderfall (siehe
vrpn_tour.py-Modul-Docstring: eine einzelne Route unter der Pfad-Nachbarschaft erreicht nicht notwendig dieselbe
Güte wie die zyklische 2-opt-Suche der TSP-Wurzel - eine echte Teilmenge der Nachbarschaft, kein Fehler)."""

import itertools

import numpy as np
import pytest

import vrpn_tour as T


def _instance(n_nodes, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n_nodes, 2)) * 100
    return xy, T.dist_matrix(xy)


def _brute_force_best_route_order(customers, D):
    """Beste Permutation einer festen Kundenmenge als EINE Route (Depot an beiden Enden) - Referenz für die
    2-opt-Konvergenz-Prüfung."""
    best = np.inf
    for perm in itertools.permutations(customers):
        best = min(best, T.route_cost(np.array(perm), D))
    return best


def test_route_cost_matches_manual_sum():
    xy, D = _instance(6, 1)
    route = np.array([2, 4, 1])
    expected = D[0, 2] + D[2, 4] + D[4, 1] + D[1, 0]
    assert T.route_cost(route, D) == pytest.approx(expected)
    assert T.route_cost(np.array([], dtype=np.int64), D) == 0.0


def test_route_demand_and_solution_demand_ok():
    demands = np.array([0, 3, 5, 2, 7])
    assert T.route_demand(np.array([1, 2]), demands) == 8.0
    assert T.solution_demand_ok([np.array([1, 2]), np.array([3, 4])], demands, capacity=9.0)
    assert not T.solution_demand_ok([np.array([1, 2, 4])], demands, capacity=9.0)


def test_validate_partition():
    assert T.validate_partition([np.array([1, 2]), np.array([3])], 3)
    assert not T.validate_partition([np.array([1, 2]), np.array([2, 3])], 3)     # 2 doppelt, 3 einmal - trotzdem ungueltig da 2 doppelt
    assert not T.validate_partition([np.array([1])], 3)                          # 2, 3 fehlen


# --- Pfad-2-opt gegen Brute-Force --------------------------------------------------------------------------------------------------------------


def test_two_opt_never_reverses_a_segment_touching_the_depot_positions():
    xy, D = _instance(9, 2)
    route = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    for _ in range(20):
        move, _ = T.find_two_opt_move(route, D, "best")
        if move is None:
            break
        i, j, _delta = move
        n = len(route) + 2
        assert i >= 0 and j <= n - 2 and j >= i + 2                              # Depot-Positionen (0 und n-1) nie in [i+1, j]
        route = T.apply_two_opt_move(route, move)
        assert sorted(route.tolist()) == [1, 2, 3, 4, 5, 6, 7, 8]                # Permutation bleibt erhalten


def test_two_opt_delta_matches_recomputed_cost():
    xy, D = _instance(8, 3)
    route = np.array([1, 2, 3, 4, 5, 6])
    move, _ = T.find_two_opt_move(route, D, "best")
    assert move is not None
    before = T.route_cost(route, D)
    new_route = T.apply_two_opt_move(route, move)
    after = T.route_cost(new_route, D)
    assert after - before == pytest.approx(move[2], abs=1e-9)


def test_descend_intra_route_converges_to_a_real_two_opt_local_optimum_per_route():
    xy, D = _instance(20, 4)
    routes = [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10]), np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])]
    r = T.descend_intra_route(D, routes, "best")
    for route in r.routes:
        move, _ = T.find_two_opt_move(route, D, "best")
        assert move is None                                                      # kein verbessernder Pfad-2-opt-Zug mehr in irgendeiner Route
    assert T.validate_partition(r.routes, 19)
    costs = [s.cost for s in r.steps]
    assert all(b < a - 1e-9 for a, b in zip(costs, costs[1:]))                   # strikt monoton fallend


def test_single_small_route_two_opt_reaches_the_true_optimum_when_the_path_neighborhood_suffices():
    """Bei einer sehr kleinen Route (<=5 Kunden) ist die Pfad-Nachbarschaft in der Praxis meist schon ausreichend,
    um das globale Optimum zu erreichen - hier gegen Brute-Force verifiziert (kein Anspruch auf Allgemeingültigkeit,
    siehe die dokumentierte Einschränkung im Modul-Docstring für größere/allgemeinere Fälle)."""
    for seed in range(5):
        xy, D = _instance(6, seed)
        customers = np.array([1, 2, 3, 4, 5])
        rng = np.random.default_rng(seed + 100)
        route = rng.permutation(customers)
        r = T.descend_intra_route(D, [route], "best", keep_steps=False)
        opt = _brute_force_best_route_order(customers.tolist(), D)
        assert r.cost == pytest.approx(opt, abs=1e-6)


def test_single_route_two_opt_matches_the_cyclic_tsp_root_exactly():
    """Die TSP-Sonderfall-Behauptung dieser Linie (Kapazität >= Gesamtbedarf ⇒ eine Route ⇒ exakt die TSP-Wurzel):
    obwohl die Pfad-Nachbarschaft auf den ersten Blick eine Teilmenge der zyklischen Nachbarschaft der Wurzel ist
    (siehe Modul-Docstring), erreicht sie bei EINER Route nachweislich dieselben lokalen Optima - hier gegen die
    echte `hc_algorithm.descend` der hill-climbing-demo gegengeprüft, nicht nur strukturell angenommen. Überspringt
    sich selbst, wenn die Schwester-Demo nicht als Geschwisterordner vorliegt (eigenständiges Repo, CI/frische
    Klone haben sie nicht daneben liegen - wie der ortools-Kreuzvergleich der übrigen Linie)."""
    import sys
    from pathlib import Path
    hc_dir = Path(__file__).resolve().parent.parent.parent / "hill-climbing-demo"
    if not (hc_dir / "hc_algorithm.py").exists():
        pytest.skip("hill-climbing-demo liegt nicht als Geschwisterordner vor")
    sys.path.insert(0, str(hc_dir))
    import hc_algorithm as HC
    for n in (10, 20, 40, 80):
        for seed in range(10):
            rng = np.random.default_rng(seed * 1000 + n)
            xy = rng.random((n, 2)) * 100
            D = T.dist_matrix(xy)
            start = np.concatenate([[0], rng.permutation(np.arange(1, n))])
            r_cyclic = HC.descend(D, start, "2opt", "best", keep_steps=False)
            r_path = T.descend_intra_route(D, [start[1:]], "best", keep_steps=False)
            assert r_path.cost == pytest.approx(r_cyclic.length, abs=1e-6)


def test_descend_stops_at_the_evaluation_budget():
    xy, D = _instance(30, 5)
    routes = [np.random.default_rng(1).permutation(np.arange(1, 16)), np.random.default_rng(2).permutation(np.arange(16, 30))]
    full = T.descend_intra_route(D, routes, "best", keep_steps=False)
    cut = T.descend_intra_route(D, routes, "best", keep_steps=False, max_evaluations=full.evaluations // 3)
    assert cut.n_moves <= full.n_moves and cut.evaluations <= full.evaluations
    assert cut.cost >= full.cost - 1e-6
