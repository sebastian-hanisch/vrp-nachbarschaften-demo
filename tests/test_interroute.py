"""Die zentrale Korrektheits-Kette der Inter-Route-Züge: jede Delta-Formel gegen direkte Kostenneuberechnung
kreuzgeprüft (nicht nur der Formel vertraut), Kapazitäts-Machbarkeit nie verletzt, die behauptete Verallgemeinerung
CROSS-exchange ⊇ {Swap, 2-opt*} tatsächlich bewiesen (nicht nur behauptet), gültige Partition und monotoner
Abstieg der kombinierten Suche, der TSP-Sonderfall (sehr große Kapazität ⇒ eine Route ⇒ dieselbe 2-opt-Lösung wie
die TSP-Wurzel dieser Linie)."""

import numpy as np
import pytest

import vrpn_construction as C
import vrpn_interroute as IR
import vrpn_scenario as S
import vrpn_tour as T


def _routes_instance(n, seed, capacity, min_routes=2):
    inst = S.generate(n, 0, seed, capacity=capacity)
    D = T.dist_matrix(inst.xy)
    routes = C.savings_construction(n, D, inst.demands, inst.capacity)
    return inst, D, routes


# --- Jede Delta-Formel gegen direkte Kostenneuberechnung ---------------------------------------------------------------------------------------


def test_relocate_delta_matches_recomputed_cost():
    inst, D, routes = _routes_instance(30, 1, capacity=40)
    for _ in range(5):
        move, _ = IR.find_relocate_move(routes, D, inst.demands, inst.capacity)
        if move is None:
            break
        before = T.solution_cost(routes, D)
        new_routes = IR.apply_relocate_move(routes, move)
        after = T.solution_cost(new_routes, D)
        assert after - before == pytest.approx(move[-1], abs=1e-6)
        assert T.solution_demand_ok(new_routes, inst.demands, inst.capacity)
        assert T.validate_partition(new_routes, 30)
        routes = new_routes


def test_swap_delta_matches_recomputed_cost():
    inst, D, routes = _routes_instance(30, 2, capacity=40)
    for _ in range(5):
        move, _ = IR.find_swap_move(routes, D, inst.demands, inst.capacity)
        if move is None:
            break
        before = T.solution_cost(routes, D)
        new_routes = IR.apply_swap_move(routes, move)
        after = T.solution_cost(new_routes, D)
        assert after - before == pytest.approx(move[-1], abs=1e-6)
        assert T.solution_demand_ok(new_routes, inst.demands, inst.capacity)
        assert T.validate_partition(new_routes, 30)
        routes = new_routes


def test_two_opt_star_delta_matches_recomputed_cost():
    inst, D, routes = _routes_instance(30, 3, capacity=40)
    for _ in range(5):
        move, _ = IR.find_two_opt_star_move(routes, D, inst.demands, inst.capacity)
        if move is None:
            break
        before = T.solution_cost(routes, D)
        new_routes = IR.apply_two_opt_star_move(routes, move)
        after = T.solution_cost(new_routes, D)
        assert after - before == pytest.approx(move[-1], abs=1e-6)
        assert T.solution_demand_ok(new_routes, inst.demands, inst.capacity)
        assert T.validate_partition(new_routes, 30)
        routes = new_routes


def test_cross_exchange_delta_matches_recomputed_cost():
    inst, D, routes = _routes_instance(30, 4, capacity=40)
    for _ in range(5):
        move, _ = IR.find_cross_exchange_move(routes, D, inst.demands, inst.capacity, max_segment=3)
        if move is None:
            break
        before = T.solution_cost(routes, D)
        new_routes = IR.apply_cross_exchange_move(routes, move)
        after = T.solution_cost(new_routes, D)
        assert after - before == pytest.approx(move[-1], abs=1e-6)
        assert T.solution_demand_ok(new_routes, inst.demands, inst.capacity)
        assert T.validate_partition(new_routes, 30)
        routes = new_routes


# --- Kapazitäts-Machbarkeit über viele Zufallsinstanzen (nie verletzt) --------------------------------------------------------------------------


def test_relocate_capacity_feasibility_over_many_instances():
    for seed in range(15):
        inst, D, routes = _routes_instance(25, seed, capacity=25)
        move, _ = IR.find_relocate_move(routes, D, inst.demands, inst.capacity)
        if move is None:
            continue
        new_routes = IR.apply_relocate_move(routes, move)
        assert T.solution_demand_ok(new_routes, inst.demands, inst.capacity)


def test_swap_capacity_feasibility_over_many_instances():
    for seed in range(15):
        inst, D, routes = _routes_instance(25, seed, capacity=25)
        move, _ = IR.find_swap_move(routes, D, inst.demands, inst.capacity)
        if move is None:
            continue
        new_routes = IR.apply_swap_move(routes, move)
        assert T.solution_demand_ok(new_routes, inst.demands, inst.capacity)


def test_two_opt_star_capacity_feasibility_over_many_instances():
    for seed in range(15):
        inst, D, routes = _routes_instance(25, seed, capacity=25)
        move, _ = IR.find_two_opt_star_move(routes, D, inst.demands, inst.capacity)
        if move is None:
            continue
        new_routes = IR.apply_two_opt_star_move(routes, move)
        assert T.solution_demand_ok(new_routes, inst.demands, inst.capacity)


def test_cross_exchange_capacity_feasibility_over_many_instances():
    for seed in range(15):
        inst, D, routes = _routes_instance(25, seed, capacity=25)
        move, _ = IR.find_cross_exchange_move(routes, D, inst.demands, inst.capacity, 3)
        if move is None:
            continue
        new_routes = IR.apply_cross_exchange_move(routes, move)
        assert T.solution_demand_ok(new_routes, inst.demands, inst.capacity)


# --- CROSS-exchange verallgemeinert Swap und 2-opt* WIRKLICH (nicht nur behauptet) -----------------------------------------------------------


def _cross_delta_at(route_a, route_b, p1, l1, p2, l2, D):
    seg_a, seg_b = route_a[p1:p1 + l1], route_b[p2:p2 + l2]
    prev_a = route_a[p1 - 1] if p1 > 0 else 0
    after_a = route_a[p1 + l1] if p1 + l1 < len(route_a) else 0
    prev_b = route_b[p2 - 1] if p2 > 0 else 0
    after_b = route_b[p2 + l2] if p2 + l2 < len(route_b) else 0
    old = D[prev_a, seg_a[0]] + D[seg_a[-1], after_a] + D[prev_b, seg_b[0]] + D[seg_b[-1], after_b]
    new = D[prev_a, seg_b[0]] + D[seg_b[-1], after_a] + D[prev_b, seg_a[0]] + D[seg_a[-1], after_b]
    return new - old


def test_cross_exchange_at_segment_length_one_equals_swap_at_every_position():
    inst, D, _routes = _routes_instance(20, 5, capacity=1000)
    route_a, route_b = np.arange(1, 6), np.arange(6, 11)
    for p in range(len(route_a)):
        for q in range(len(route_b)):
            cross_delta = _cross_delta_at(route_a, route_b, p, 1, q, 1, D)
            c1, c2 = route_a[p], route_b[q]
            prev_a = route_a[p - 1] if p > 0 else 0
            next_a = route_a[p + 1] if p + 1 < len(route_a) else 0
            prev_b = route_b[q - 1] if q > 0 else 0
            next_b = route_b[q + 1] if q + 1 < len(route_b) else 0
            old = D[prev_a, c1] + D[c1, next_a] + D[prev_b, c2] + D[c2, next_b]
            new = D[prev_a, c2] + D[c2, next_a] + D[prev_b, c1] + D[c1, next_b]
            swap_delta = new - old
            assert cross_delta == pytest.approx(swap_delta, abs=1e-9)


def test_cross_exchange_to_route_end_equals_two_opt_star_at_every_position():
    inst, D, _routes = _routes_instance(20, 6, capacity=1000)
    route_a, route_b = np.arange(1, 6), np.arange(6, 11)
    for i in range(len(route_a)):
        for j in range(len(route_b)):
            cross_delta = _cross_delta_at(route_a, route_b, i, len(route_a) - i, j, len(route_b) - j, D)
            end_a = route_a[i - 1] if i > 0 else 0
            start_a_tail = route_a[i]
            end_b = route_b[j - 1] if j > 0 else 0
            start_b_tail = route_b[j]
            star_delta = (D[end_a, start_b_tail] + D[end_b, start_a_tail]) - (D[end_a, start_a_tail] + D[end_b, start_b_tail])
            assert cross_delta == pytest.approx(star_delta, abs=1e-9)


# --- Kombinierte Suche: gültige Partition, Kapazität, monotoner Abstieg -------------------------------------------------------------------------


def test_combined_descend_stays_feasible_and_valid_and_monotone():
    inst, D, routes = _routes_instance(40, 7, capacity=40)
    r = IR.descend(D, routes, inst.demands, inst.capacity, keep_steps=True)
    assert T.solution_demand_ok(r.routes, inst.demands, inst.capacity)
    assert T.validate_partition(r.routes, 40)
    costs = [s.cost for s in r.steps]
    assert all(b <= a + 1e-6 for a, b in zip(costs, costs[1:]))


def test_combined_descend_never_worse_than_construction():
    for seed in range(8):
        inst, D, routes = _routes_instance(30, seed, capacity=30)
        before = T.solution_cost(routes, D)
        r = IR.descend(D, routes, inst.demands, inst.capacity, keep_steps=False)
        assert r.cost <= before + 1e-6


def test_active_moves_toggle_restricts_which_kinds_are_used():
    inst, D, routes = _routes_instance(40, 8, capacity=40)
    r = IR.descend(D, routes, inst.demands, inst.capacity, active_moves=("relocate",), keep_steps=False)
    assert set(r.kinds) <= {"relocate"}


def test_descend_stops_at_the_evaluation_budget():
    inst, D, routes = _routes_instance(40, 9, capacity=40)
    full = IR.descend(D, routes, inst.demands, inst.capacity, keep_steps=False)
    cut = IR.descend(D, routes, inst.demands, inst.capacity, keep_steps=False, max_evaluations=max(full.evaluations // 3, 1))
    assert cut.n_moves <= full.n_moves and cut.cost >= full.cost - 1e-6


# --- TSP-Sonderfall: sehr große Kapazität ⇒ eine Route ⇒ dieselbe 2-opt-Lösung wie die TSP-Wurzel --------------------------------------------


def test_large_capacity_pipeline_matches_the_tsp_root_two_opt_descent_exactly():
    """Überspringt sich selbst ohne den Geschwisterordner hill-climbing-demo (eigenständiges Repo, CI/frische
    Klone haben ihn nicht daneben liegen)."""
    import sys
    from pathlib import Path
    hc_dir = Path(__file__).resolve().parent.parent.parent / "hill-climbing-demo"
    if not (hc_dir / "hc_algorithm.py").exists():
        pytest.skip("hill-climbing-demo liegt nicht als Geschwisterordner vor")
    sys.path.insert(0, str(hc_dir))
    import hc_algorithm as HC
    for seed in range(5):
        inst, D, routes = _routes_instance(40, seed, capacity=100000)
        r = IR.descend(D, routes, inst.demands, inst.capacity, keep_steps=False)
        assert len(r.routes) == 1
        start = np.concatenate([[0]] + [rt for rt in routes])
        ref = HC.descend(D, start, "2opt", "best", keep_steps=False)
        assert r.cost == pytest.approx(ref.length, abs=1e-6)
