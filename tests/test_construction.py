"""Clarke & Wright Savings-Konstruktion: Kapazität nie verletzt, gültige Partition, nie schlechter als "eine Route
je Kunde", Routenzahl sinkt mit wachsender Kapazität, TSP-Sonderfall bei sehr großer Kapazität."""

import numpy as np
import pytest

import vrpn_construction as C
import vrpn_scenario as S
import vrpn_tour as T


def test_savings_never_violates_capacity_and_produces_a_valid_partition():
    for seed in range(10):
        inst = S.generate(40, 0, seed, capacity=40)
        D = T.dist_matrix(inst.xy)
        routes = C.savings_construction(40, D, inst.demands, inst.capacity)
        assert T.solution_demand_ok(routes, inst.demands, inst.capacity)
        assert T.validate_partition(routes, 40)


def test_savings_is_never_worse_than_one_route_per_customer():
    inst = S.generate(30, 0, 3, capacity=25)
    D = T.dist_matrix(inst.xy)
    routes = C.savings_construction(30, D, inst.demands, inst.capacity)
    naive_cost = sum(2 * D[0, c] for c in range(1, 31))                          # ein Fahrzeug pro Kunde, hin und zurück
    assert T.solution_cost(routes, D) <= naive_cost + 1e-6


def test_route_count_shrinks_as_capacity_grows():
    inst_base = S.generate(60, 0, 35, capacity=1)
    D = T.dist_matrix(inst_base.xy)
    counts = []
    for cap in (15, 30, 60, 120, 600):
        inst = S.generate(60, 0, 35, capacity=cap)
        routes = C.savings_construction(60, D, inst.demands, cap)
        counts.append(len(routes))
    assert counts == sorted(counts, reverse=True)                                # nie steigend mit wachsender Kapazität
    assert counts[0] > counts[-1]                                                # echter Unterschied zwischen den Extremen


def test_very_large_capacity_collapses_to_a_single_route():
    for seed in range(10):
        inst = S.generate(30, 0, seed, capacity=100000)
        D = T.dist_matrix(inst.xy)
        routes = C.savings_construction(30, D, inst.demands, 100000)
        assert len(routes) == 1
        assert sorted(routes[0].tolist()) == list(range(1, 31))


def test_try_merge_respects_endpoints_and_reverses_when_needed():
    assert C._try_merge([1, 2, 3], [4, 5], 3, 4) == [1, 2, 3, 4, 5]
    assert C._try_merge([1, 2, 3], [4, 5], 1, 5) == [3, 2, 1, 5, 4]
    assert C._try_merge([1, 2, 3], [4, 5], 2, 4) is None                          # 2 ist kein Ende von [1,2,3]
