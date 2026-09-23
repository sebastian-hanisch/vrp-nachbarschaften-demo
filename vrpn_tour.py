"""Routendarstellung, Kosten/Machbarkeit und Intra-Route-2-opt für die VRP-Nachbarschaften-Demo.

Eine Route ist ein 1D-Array von Kundenindizes (OHNE Depot); eine Lösung ist eine Liste von Routen. Depot (Index 0)
ist an beiden Enden JEDER Route implizit. Anders als die TSP-Wurzel dieser Linie (`hill-climbing-demo`, EIN
geschlossener Zyklus durch ALLE Knoten) ist eine Route hier ein OFFENER PFAD: Depot -> c1 -> ... -> ck -> Depot,
mit dem Depot an BEIDEN Enden fest verankert (es gehört gleichzeitig zu jeder anderen Route und kann deshalb nicht
selbst "gedreht" werden, wie es eine zyklische 2-opt-Formel bei einer einzelnen Tour täte).

Das intra-route 2-opt hier ist deshalb eine EIGENSTÄNDIG hergeleitete, nicht einfach wiederverwendete Variante von
`hc_algorithm._delta_2opt`: die Gültigkeitsbedingung schließt JEDE Umkehrung aus, die eine der beiden
Depot-Positionen (Anfang oder Ende der Route) mit einschließen würde (`j <= n-2` statt nur des einzelnen
Sonderfalls `(i=0, j=n-1)`, den die zyklische Formel der Wurzel ausschließt).

Auf den ersten Blick sieht das nach einer ECHTEN Einschränkung aus (eine Teilmenge der zyklischen Nachbarschaft,
die keine Umkehrung "durch" das Depot hindurch ausdrücken kann) - tatsächlich ist sie es aber NICHT: jede
zyklische 2-opt-Umkehrung [i+1, j] hat eine KOMPLEMENTÄRE Darstellung (die Umkehrung des Rests der Tour), die
exakt dieselbe resultierende Tour ergibt; die einzigen Paare, die die Pfad-Bedingung `j <= n-2` ausschließt (mit
j = n-1), sind IMMER redundant zu einem bereits eingeschlossenen Paar. Die Pfad-Nachbarschaft erreicht deshalb im
TSP-Sonderfall (genau eine Route, Kapazität >= Gesamtbedarf) NACHWEISLICH dieselben lokalen Optima wie die
zyklische 2-opt-Suche der TSP-Wurzel - empirisch verifiziert (80 Zufallsinstanzen, n=10..80, siehe
tests/test_tour.py::test_single_route_two_opt_matches_the_cyclic_tsp_root_exactly, Differenz in JEDEM Fall exakt
0), nicht nur vermutet.

Intra-Route-Or-opt ist bewusst NICHT separat implementiert - es ist der Spezialfall des CROSS-exchange-Zugs
(vrpn_interroute.py) mit gleicher Quell- und Zielroute; ein eigenständiges Modul dafür wäre reine Duplikation."""

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


def dist_matrix(xy):
    xy = np.asarray(xy, dtype=float)
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


def route_cost(route, D):
    if len(route) == 0:
        return 0.0
    t = np.concatenate([[0], route, [0]])
    return float(D[t[:-1], t[1:]].sum())


def route_demand(route, demands):
    return float(demands[route].sum()) if len(route) else 0.0


def solution_cost(routes, D):
    return sum(route_cost(r, D) for r in routes)


def solution_demand_ok(routes, demands, capacity):
    return all(route_demand(r, demands) <= capacity + EPS for r in routes)


def validate_partition(routes, n):
    """Jeder Kunde 1..n muss in GENAU einer Route vorkommen."""
    seen = np.concatenate(routes) if routes else np.array([], dtype=np.int64)
    return sorted(seen.tolist()) == list(range(1, n + 1))


# --- Intra-Route 2-opt (Pfad, Depot an beiden Enden fest) --------------------------------------------------------------------------------------


def _valid_pairs_path(n):
    """Paare i < j mit j >= i+2, UND j <= n-2 (Position n-1 - das zweite Depot-Ende - darf nie Teil der
    umgekehrten Strecke [i+1, j] sein; Position 0 - das erste Depot-Ende - liegt als `i` selbst nie in der
    umgekehrten Strecke und ist deshalb unproblematisch)."""
    i, j = np.indices((n, n))
    return (j >= i + 2) & (j <= n - 2)


def _delta_2opt_path(t, D):
    """`t` ist die Route MIT Depot an beiden Enden (`[0, c1, ..., ck, 0]`, Länge n)."""
    n = len(t)
    nxt = np.roll(t, -1)
    e = D[t, nxt]
    delta = D[np.ix_(t, t)] + D[np.ix_(nxt, nxt)] - e[:, None] - e[None, :]
    return delta, _valid_pairs_path(n)


def find_two_opt_move(route, D, rule="best"):
    """Bestes/erstes verbesserndes 2-opt innerhalb EINER Route (Depot an beiden Enden fest). Gibt ((i, j, delta)
    oder None, Zahl bewerteter Nachbarn) zurück - Positionen beziehen sich auf `t = [0, *route, 0]`."""
    if len(route) < 3:
        return None, 0
    t = np.concatenate([[0], route, [0]])
    delta, ok = _delta_2opt_path(t, D)
    d = np.where(ok, delta, np.inf)
    flat = d.ravel()
    n = len(t)
    if rule == "best":
        k = int(np.argmin(flat))
        if flat[k] >= -EPS:
            return None, int(ok.sum())
        i, j = divmod(k, n)
        return (i, j, float(flat[k])), int(ok.sum())
    improving = np.where(flat < -EPS)[0]
    if len(improving) == 0:
        return None, int(ok.sum())
    k = int(improving[0])
    i, j = divmod(k, n)
    return (i, j, float(flat[k])), int(ok.sum())


def apply_two_opt_move(route, move):
    i, j, _delta = move
    t = np.concatenate([[0], route, [0]])
    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
    return t[1:-1]


@dataclass
class Step:
    move: object
    routes: list
    cost: float


@dataclass
class Descent:
    routes: list
    cost: float
    steps: list = field(default_factory=list)
    evaluations: int = 0
    n_moves: int = 0


def descend_intra_route(D, routes, rule="best", max_moves=100000, keep_steps=True, max_evaluations=None):
    """2-opt je Route für sich, bis keine Route mehr eine Verbesserung findet (lokales Optimum unter der reinen
    Intra-Route-Nachbarschaft), `max_moves` Züge insgesamt oder das Bewertungsbudget erreicht sind."""
    routes = [r.copy() for r in routes]
    cost = solution_cost(routes, D)
    steps = [Step(None, [r.copy() for r in routes], cost)] if keep_steps else []
    evaluations = n_moves = 0
    for _ in range(max_moves):
        if max_evaluations is not None and evaluations >= max_evaluations:
            break
        best = None
        for v, route in enumerate(routes):
            move, count = find_two_opt_move(route, D, rule)
            evaluations += count
            if move is not None and (best is None or move[2] < best[1][2] - EPS):
                best = (v, move)
        if best is None:
            break
        v, move = best
        routes[v] = apply_two_opt_move(routes[v], move)
        cost += move[2]
        n_moves += 1
        if keep_steps:
            steps.append(Step(("2opt", v, move), [r.copy() for r in routes], cost))
    cost = solution_cost(routes, D)
    return Descent(routes, cost, steps, evaluations, n_moves)
