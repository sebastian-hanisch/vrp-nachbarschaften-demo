"""Orakel-Test für die Inter-Route-Nachbarschaften: Jede Zugart wird hier durch explizites Umbauen der Routenlisten und VOLLSTÄNDIGES Neurechnen
der Lösungskosten enumeriert (nicht über die Delta-Formeln des Codes), Zulässigkeit (Kapazität, Partition) durch Nachzählen. Verglichen werden
bestes Delta, Bewertungszähler (eigene Zählformel je Zugart), Anwendung des Zugs und das Ergebnis des Abstiegs (lokales Optimum aller aktiven
Zugarten, Routenliste ohne leere Routen). Die Savings-Konstruktion wird gegen ihre definierende Invariante geprüft (keine verschmelzbaren
Endpunktpaare mit positiver Ersparnis, die in die Kapazität passen)."""

import numpy as np

import vrpn_construction as CO
import vrpn_interroute as IR
import vrpn_scenario as S
import vrpn_tour as T

EPS = 1e-9


def _cost(routes, Dl):
    total = 0.0
    for r in routes:
        if len(r):
            seq = [0, *[int(c) for c in r], 0]
            total += sum(Dl[a][b] for a, b in zip(seq, seq[1:]))
    return total


def _feasible(routes, demands, capacity):
    return all(sum(int(demands[c]) for c in r) <= capacity + EPS for r in routes)


def _lists(routes):
    return [[int(c) for c in r] for r in routes]


def _random_solution(rng, n, demands, capacity, k):
    """Zulässige Zufallslösung: Kunden gemischt, in k Routen verteilt, die Kapazität einhält (sonst neue Route)."""
    order = [int(c) for c in rng.permutation(np.arange(1, n + 1))]
    routes, loads = [[] for _ in range(k)], [0.0] * k
    for c in order:
        cands = [v for v in range(k) if loads[v] + demands[c] <= capacity]
        if not cands:
            routes.append([])
            loads.append(0.0)
            cands = [len(routes) - 1]
        v = int(rng.choice(cands))
        routes[v].append(c)
        loads[v] += demands[c]
    return [np.array(r, dtype=np.int64) for r in routes if r]


def _instance(rng):
    n = int(rng.integers(4, 10))
    cap = float(rng.integers(9, 30))
    inst = S.generate(n, int(rng.choice([0, 50, 100])), int(rng.integers(0, 10**6)), capacity=cap)
    xy = inst.xy
    if rng.random() < 0.3:                                          # Gleichstände: Punkte auf grobem Gitter
        xy = rng.integers(0, 4, (n + 1, 2)).astype(float)
        xy[0] = (1.5, 1.5)
    return inst, T.dist_matrix(xy)


def _enumerate(routes, kind, max_segment):
    """Alle Nachbarlösungen (neue Routenlisten) der Zugart `kind`, explizit konstruiert; Quell- und Zielroute verschieden."""
    r = _lists(routes)
    out = []
    m = len(r)
    if kind == "relocate":
        for a in range(m):
            for p in range(len(r[a])):
                for b in range(m):
                    if b == a:
                        continue
                    for q in range(len(r[b]) + 1):
                        new = [x[:] for x in r]
                        c = new[a].pop(p)
                        new[b].insert(q, c)
                        out.append(new)
    elif kind == "swap":
        for a in range(m):
            for b in range(a + 1, m):
                for p in range(len(r[a])):
                    for q in range(len(r[b])):
                        new = [x[:] for x in r]
                        new[a][p], new[b][q] = new[b][q], new[a][p]
                        out.append(new)
    elif kind == "2opt_star":
        for a in range(m):
            for b in range(a + 1, m):
                for i in range(len(r[a]) + 1):
                    for j in range(len(r[b]) + 1):
                        if (i == 0 and j == 0) or (i == len(r[a]) and j == len(r[b])):
                            continue
                        new = [x[:] for x in r]
                        new[a], new[b] = r[a][:i] + r[b][j:], r[b][:j] + r[a][i:]
                        out.append(new)
    elif kind == "cross":
        for a in range(m):
            for b in range(a + 1, m):
                for l1 in range(1, min(max_segment, len(r[a])) + 1):
                    for p1 in range(len(r[a]) - l1 + 1):
                        for l2 in range(1, min(max_segment, len(r[b])) + 1):
                            for p2 in range(len(r[b]) - l2 + 1):
                                new = [x[:] for x in r]
                                new[a] = r[a][:p1] + r[b][p2:p2 + l2] + r[a][p1 + l1:]
                                new[b] = r[b][:p2] + r[a][p1:p1 + l1] + r[b][p2 + l2:]
                                out.append(new)
    elif kind == "intra":
        for a in range(m):
            t = [0, *r[a], 0]
            for i in range(len(t)):
                for j in range(i + 2, len(t) - 1):                  # umgekehrt wird t[i+1..j]; die Depot-Enden bleiben fest
                    new = [x[:] for x in r]
                    new[a] = (t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:])[1:-1]
                    out.append(new)
    return out


def _expected_evaluations(routes, kind, max_segment):
    r = [len(x) for x in routes]
    m = len(r)
    if kind == "relocate":
        return sum(r[a] * sum(r[b] + 1 for b in range(m) if b != a) for a in range(m))
    if kind == "swap":
        return sum(r[a] * r[b] for a in range(m) for b in range(a + 1, m))
    if kind == "2opt_star":
        return sum((r[a] + 1) * (r[b] + 1) - 2 for a in range(m) for b in range(a + 1, m))
    if kind == "cross":
        return sum((r[a] - l1 + 1) * (r[b] - l2 + 1) for a in range(m) for b in range(a + 1, m)
                   for l1 in range(1, max_segment + 1) for l2 in range(1, max_segment + 1) if l1 <= r[a] and l2 <= r[b])
    raise ValueError(kind)


FINDERS = {"relocate": (IR.find_relocate_move, IR.apply_relocate_move), "swap": (IR.find_swap_move, IR.apply_swap_move),
           "2opt_star": (IR.find_two_opt_star_move, IR.apply_two_opt_star_move)}


def test_every_neighbourhood_finds_the_best_improving_neighbour_of_an_explicit_enumeration():
    rng = np.random.default_rng(2025)
    improving_found = 0
    for _ in range(220):
        inst, D = _instance(rng)
        Dl = D.tolist()
        routes = _random_solution(rng, inst.n, inst.demands, inst.capacity, int(rng.integers(1, 4)))
        base = _cost(routes, Dl)
        for kind in ("relocate", "swap", "2opt_star", "cross"):
            if kind == "cross":
                move, count = IR.find_cross_exchange_move(routes, D, inst.demands, inst.capacity, 3)
                apply = IR.apply_cross_exchange_move
            else:
                move, count = FINDERS[kind][0](routes, D, inst.demands, inst.capacity)
                apply = FINDERS[kind][1]
            neighbours = [x for x in _enumerate(routes, kind, 3) if _feasible(x, inst.demands, inst.capacity)]
            best = min((_cost(x, Dl) - base for x in neighbours), default=None)
            assert count == _expected_evaluations(routes, kind, 3), kind
            if best is not None and best < -EPS:
                assert move is not None and abs(move[-1] - best) < 1e-7, (kind, move, best)
                new = apply(routes, move)
                assert abs(_cost(new, Dl) - base - move[-1]) < 1e-7
                assert _feasible(new, inst.demands, inst.capacity) and T.validate_partition(new, inst.n)
                improving_found += 1
            else:
                assert move is None, (kind, move, best)
    assert improving_found > 150


def test_intra_route_two_opt_matches_an_explicit_enumeration_and_the_evaluation_count():
    rng = np.random.default_rng(8)
    found = 0
    for _ in range(250):
        n = int(rng.integers(1, 9))
        xy = rng.random((n + 1, 2)) * 50 if rng.random() < 0.6 else rng.integers(0, 4, (n + 1, 2)).astype(float)
        D = T.dist_matrix(xy)
        Dl = D.tolist()
        route = rng.permutation(np.arange(1, n + 1)).astype(np.int64)
        base = _cost([route], Dl)
        move, count = T.find_two_opt_move(route, D, "best")
        neighbours = _enumerate([route], "intra", 3)
        best = min((_cost(x, Dl) - base for x in neighbours), default=None)
        if len(route) < 3:
            assert move is None and count == 0                                 # Umkehr einer 2er-Route ändert bei symmetrischen Abständen nichts
            assert all(abs(_cost(x, Dl) - base) < 1e-9 for x in neighbours)
            continue
        assert count == len(neighbours)
        if best is not None and best < -EPS:
            assert move is not None and abs(move[2] - best) < 1e-7
            assert abs(_cost([T.apply_two_opt_move(route, move)], Dl) - base - move[2]) < 1e-7
            found += 1
        else:
            assert move is None
    assert found > 60


def test_descend_ends_in_a_local_optimum_of_all_active_moves_with_exact_cost_and_a_clean_route_list():
    rng = np.random.default_rng(99)
    for _ in range(40):
        inst, D = _instance(rng)
        Dl = D.tolist()
        routes = _random_solution(rng, inst.n, inst.demands, inst.capacity, int(rng.integers(1, 5)))
        start = _cost(routes, Dl)
        active = tuple(k for k in IR.MOVE_TYPES if rng.random() < 0.7) or ("intra",)
        r = IR.descend(D, routes, inst.demands, inst.capacity, active_moves=active, keep_steps=False)
        assert abs(r.cost - _cost(r.routes, Dl)) < 1e-7 and r.cost <= start + 1e-9
        assert _feasible(r.routes, inst.demands, inst.capacity) and T.validate_partition(r.routes, inst.n)
        assert all(len(x) > 0 for x in r.routes)                          # keine leeren "Routen"
        for kind in active:
            assert not any(_cost(x, Dl) < r.cost - 1e-7 for x in _enumerate(r.routes, kind, 3) if _feasible(x, inst.demands, inst.capacity)), kind
        assert sum(r.kinds.values()) == r.n_moves


def test_a_move_that_empties_a_route_does_not_leave_an_empty_route_in_the_result():
    """Zwei Einzelkunden-Routen, die fast auf einem Strahl liegen: der Kunde der kürzeren Route wandert in die andere ein, die Route verschwindet."""
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 1.0]])
    D = T.dist_matrix(xy)
    demands = np.array([0, 1, 1])
    routes = [np.array([1]), np.array([2])]
    r = IR.descend(D, routes, demands, 10.0, active_moves=("relocate",), keep_steps=False)
    assert len(r.routes) == 1 and sorted(r.routes[0].tolist()) == [1, 2]
    assert abs(r.cost - _cost([[1, 2]], D.tolist())) < 1e-9 and abs(r.cost - (10 + 1 + np.sqrt(101))) < 1e-9


def test_savings_construction_is_feasible_a_partition_and_has_no_mergeable_endpoint_pair_left():
    rng = np.random.default_rng(4)
    for _ in range(150):
        n = int(rng.integers(1, 12))
        cap = float(rng.integers(9, 40))
        inst = S.generate(n, int(rng.choice([0, 100])), int(rng.integers(0, 10**6)), capacity=cap)
        D = T.dist_matrix(inst.xy)
        routes = [x.tolist() for x in CO.savings_construction(n, D, inst.demands, cap)]
        assert sorted(c for x in routes for c in x) == list(range(1, n + 1))
        assert all(sum(int(inst.demands[c]) for c in x) <= cap for x in routes)
        loads = [sum(int(inst.demands[c]) for c in x) for x in routes]
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                if loads[a] + loads[b] > cap:
                    continue
                for i in {routes[a][0], routes[a][-1]}:
                    for j in {routes[b][0], routes[b][-1]}:
                        assert D[i, 0] + D[0, j] - D[i, j] <= 1e-9, (routes, i, j)       # sonst hätte Clarke & Wright sie verschmolzen
