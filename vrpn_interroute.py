"""Inter-Route-Züge - existieren erst, sobald es mehrere Routen gibt (auf einer einzelnen TSP-Tour undenkbar):

- **Relocate** (Shift(1,0)): ein Kunde wandert von Route A an eine beliebige Position in Route B.
- **Swap(1,1)**: ein Kunde aus Route A tauscht Platz mit einem Kunden aus Route B.
- **2-opt\\*** (Potvin & Rousseau 1995, *An Exchange Heuristic for Routeing Problems with Time Windows*, JORS
  46(12), 1433-1446): die ENDEN zweier Routen ab je einer Position werden vertauscht (Route A bis Position i +
  Route B ab Position j, und umgekehrt) - verändert oft viele Kunden auf einmal.
- **CROSS-exchange** (Taillard, Badeau, Gendreau, Guertin & Potvin 1997, *A Tabu Search Heuristic for the VRP
  with Time Windows*, Transportation Science 31(2), 170-186): ein Segment (Länge 1..MAX_SEGMENT, feste
  Reihenfolge - keine Segment-Umkehrung, siehe Grenzen) wird zwischen zwei Routen getauscht. Beweisbar (nicht nur
  behauptet) die Verallgemeinerung der drei anderen Züge: Segmentlänge 1/1 == Swap, Segmentlänge bis Routenende auf
  beiden Seiten == 2-opt\\* (beide gegen die eigenständige Herleitung von `find_two_opt_star_move`/`find_swap_move`
  regressionsgeprüft, siehe tests/test_interroute.py).

Jede Delta-Formel liest nur die vier (bei CROSS-exchange: die Segment-Ränder, ebenfalls konstant viele) betroffenen
Kanten - kein voller Neuberechnungs-Durchlauf je Kandidat -, ist aber gegen eine direkte Kostenneuberechnung
(vorher/nachher) kreuzgeprüft, nicht nur der Formel vertraut (siehe tests/test_interroute.py, zentral)."""

from dataclasses import dataclass, field

import numpy as np

import vrpn_tour as T

EPS = 1e-9
MOVE_TYPES = ("intra", "relocate", "swap", "2opt_star", "cross")
MOVE_LABELS = {"intra": "Intra-Route (2-opt)", "relocate": "Relocate", "swap": "Swap", "2opt_star": "2-opt*", "cross": "CROSS-exchange"}


# --- Relocate ----------------------------------------------------------------------------------------------------------------------------------


def find_relocate_move(routes, D, demands, capacity):
    """Bestes Relocate: ein Kunde wandert von einer Route in eine ANDERE. Gibt ((v_from, p, v_to, q, delta) oder
    None, Zahl bewerteter Nachbarn) zurück - `p` ist die Position in `routes[v_from]`, `q` die Zielposition
    (0..len(routes[v_to])) in `routes[v_to]` NACH dem Entfernen."""
    best = None
    evaluations = 0
    loads = [float(demands[r].sum()) if len(r) else 0.0 for r in routes]
    for v_from, route_from in enumerate(routes):
        for p in range(len(route_from)):
            c = route_from[p]
            prev = route_from[p - 1] if p > 0 else 0
            nxt = route_from[p + 1] if p + 1 < len(route_from) else 0
            removal_gain = D[prev, c] + D[c, nxt] - D[prev, nxt]
            for v_to, route_to in enumerate(routes):
                if v_to == v_from:
                    continue
                if loads[v_to] + demands[c] > capacity + EPS:
                    evaluations += len(route_to) + 1
                    continue
                for q in range(len(route_to) + 1):
                    prev_b = route_to[q - 1] if q > 0 else 0
                    next_b = route_to[q] if q < len(route_to) else 0
                    evaluations += 1
                    insert_cost = D[prev_b, c] + D[c, next_b] - D[prev_b, next_b]
                    delta = insert_cost - removal_gain
                    if delta < -EPS and (best is None or delta < best[4] - EPS):
                        best = (v_from, p, v_to, q, delta)
    return best, evaluations


def apply_relocate_move(routes, move):
    v_from, p, v_to, q, _delta = move
    routes = [r.copy() for r in routes]
    c = routes[v_from][p]
    routes[v_from] = np.delete(routes[v_from], p)
    routes[v_to] = np.insert(routes[v_to], q, c)
    return routes


# --- Swap(1,1) -----------------------------------------------------------------------------------------------------------------------------------


def find_swap_move(routes, D, demands, capacity):
    """Bestes Swap(1,1): je ein Kunde aus zwei VERSCHIEDENEN Routen tauscht die Position."""
    best = None
    evaluations = 0
    loads = [float(demands[r].sum()) if len(r) else 0.0 for r in routes]
    for va in range(len(routes)):
        route_a = routes[va]
        for vb in range(va + 1, len(routes)):
            route_b = routes[vb]
            for p, c1 in enumerate(route_a):
                prev_a = route_a[p - 1] if p > 0 else 0
                next_a = route_a[p + 1] if p + 1 < len(route_a) else 0
                for q, c2 in enumerate(route_b):
                    evaluations += 1
                    new_load_a = loads[va] - demands[c1] + demands[c2]
                    new_load_b = loads[vb] - demands[c2] + demands[c1]
                    if new_load_a > capacity + EPS or new_load_b > capacity + EPS:
                        continue
                    prev_b = route_b[q - 1] if q > 0 else 0
                    next_b = route_b[q + 1] if q + 1 < len(route_b) else 0
                    old = D[prev_a, c1] + D[c1, next_a] + D[prev_b, c2] + D[c2, next_b]
                    new = D[prev_a, c2] + D[c2, next_a] + D[prev_b, c1] + D[c1, next_b]
                    delta = new - old
                    if delta < -EPS and (best is None or delta < best[4] - EPS):
                        best = (va, p, vb, q, delta)
    return best, evaluations


def apply_swap_move(routes, move):
    va, p, vb, q, _delta = move
    routes = [r.copy() for r in routes]
    routes[va][p], routes[vb][q] = routes[vb][q], routes[va][p]
    return routes


# --- 2-opt* ----------------------------------------------------------------------------------------------------------------------------------


def find_two_opt_star_move(routes, D, demands, capacity):
    """Bestes 2-opt*: die Enden zweier VERSCHIEDENER Routen ab Position i (Route A) bzw. j (Route B) werden
    vertauscht - A' = A[:i] + B[j:], B' = B[:j] + A[i:]."""
    best = None
    evaluations = 0
    loads = [float(demands[r].sum()) if len(r) else 0.0 for r in routes]
    prefix = [np.concatenate([[0.0], np.cumsum(demands[r])]) if len(r) else np.array([0.0]) for r in routes]
    for va in range(len(routes)):
        route_a = routes[va]
        for vb in range(va + 1, len(routes)):
            route_b = routes[vb]
            for i in range(len(route_a) + 1):
                end_a = route_a[i - 1] if i > 0 else 0
                start_a_tail = route_a[i] if i < len(route_a) else 0
                load_a_prefix = prefix[va][i]
                for j in range(len(route_b) + 1):
                    if i == 0 and j == 0:
                        continue                                                  # no-op (beide Routen bleiben wie sie sind)
                    if i == len(route_a) and j == len(route_b):
                        continue                                                  # no-op
                    evaluations += 1
                    end_b = route_b[j - 1] if j > 0 else 0
                    start_b_tail = route_b[j] if j < len(route_b) else 0
                    new_load_a = load_a_prefix + (loads[vb] - prefix[vb][j])
                    new_load_b = prefix[vb][j] + (loads[va] - load_a_prefix)
                    if new_load_a > capacity + EPS or new_load_b > capacity + EPS:
                        continue
                    delta = (D[end_a, start_b_tail] + D[end_b, start_a_tail]) - (D[end_a, start_a_tail] + D[end_b, start_b_tail])
                    if delta < -EPS and (best is None or delta < best[4] - EPS):
                        best = (va, i, vb, j, delta)
    return best, evaluations


def apply_two_opt_star_move(routes, move):
    va, i, vb, j, _delta = move
    routes = [r.copy() for r in routes]
    a, b = routes[va], routes[vb]
    new_a = np.concatenate([a[:i], b[j:]])
    new_b = np.concatenate([b[:j], a[i:]])
    routes[va], routes[vb] = new_a, new_b
    return routes


# --- CROSS-exchange ----------------------------------------------------------------------------------------------------------------------------


def find_cross_exchange_move(routes, D, demands, capacity, max_segment):
    """Bestes CROSS-exchange: ein Segment (Länge 1..max_segment) aus Route A wird gegen ein Segment (Länge
    1..max_segment) aus Route B getauscht (feste Reihenfolge, keine Umkehrung - siehe Modul-Docstring)."""
    best = None
    evaluations = 0
    loads = [float(demands[r].sum()) if len(r) else 0.0 for r in routes]
    for va in range(len(routes)):
        route_a = routes[va]
        for vb in range(va + 1, len(routes)):
            route_b = routes[vb]
            for l1 in range(1, max_segment + 1):
                if l1 > len(route_a):
                    continue
                for p1 in range(len(route_a) - l1 + 1):
                    seg_a = route_a[p1:p1 + l1]
                    prev_a = route_a[p1 - 1] if p1 > 0 else 0
                    after_a = route_a[p1 + l1] if p1 + l1 < len(route_a) else 0
                    demand_a = float(demands[seg_a].sum())
                    for l2 in range(1, max_segment + 1):
                        if l2 > len(route_b):
                            continue
                        for p2 in range(len(route_b) - l2 + 1):
                            seg_b = route_b[p2:p2 + l2]
                            evaluations += 1
                            demand_b = float(demands[seg_b].sum())
                            new_load_a = loads[va] - demand_a + demand_b
                            new_load_b = loads[vb] - demand_b + demand_a
                            if new_load_a > capacity + EPS or new_load_b > capacity + EPS:
                                continue
                            prev_b = route_b[p2 - 1] if p2 > 0 else 0
                            after_b = route_b[p2 + l2] if p2 + l2 < len(route_b) else 0
                            old = D[prev_a, seg_a[0]] + D[seg_a[-1], after_a] + D[prev_b, seg_b[0]] + D[seg_b[-1], after_b]
                            new = D[prev_a, seg_b[0]] + D[seg_b[-1], after_a] + D[prev_b, seg_a[0]] + D[seg_a[-1], after_b]
                            delta = new - old
                            if delta < -EPS and (best is None or delta < best[6] - EPS):
                                best = (va, p1, l1, vb, p2, l2, delta)
    return best, evaluations


def apply_cross_exchange_move(routes, move):
    va, p1, l1, vb, p2, l2, _delta = move
    routes = [r.copy() for r in routes]
    a, b = routes[va], routes[vb]
    seg_a, seg_b = a[p1:p1 + l1], b[p2:p2 + l2]
    new_a = np.concatenate([a[:p1], seg_b, a[p1 + l1:]])
    new_b = np.concatenate([b[:p2], seg_a, b[p2 + l2:]])
    routes[va], routes[vb] = new_a, new_b
    return routes


# --- Kombinierte Suche (beste Verbesserung über alle aktivierten Zugarten, voller Rescan) -------------------------------------------------------


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
    kinds: dict = field(default_factory=dict)


def _best_intra(routes, D):
    """(v, inner_move) für das beste intra-route 2-opt über ALLE Routen, oder None."""
    best = None
    evaluations = 0
    for v, route in enumerate(routes):
        move, count = T.find_two_opt_move(route, D, "best")
        evaluations += count
        if move is not None and (best is None or move[2] < best[1][2] - EPS):
            best = (v, move)
    return best, evaluations


def descend(D, routes, demands, capacity, active_moves=MOVE_TYPES, max_segment=3, max_moves=100000, keep_steps=True, max_evaluations=None):
    """Bestes-Verbesserung-Suche über die KOMBINIERTE Kandidatenmenge aller in `active_moves` aktivierten
    Zugarten (voller Rescan je Iteration, "ein Vorschlag = ein bewerteter Nachbar" - dieselbe Konvention wie jedes
    Stück dieser Linie). `active_moves` steuert die Ablations-Messreihe (Grenler in der App)."""
    routes = [r.copy() for r in routes]
    cost = T.solution_cost(routes, D)
    steps = [Step(None, [r.copy() for r in routes], cost)] if keep_steps else []
    evaluations = n_moves = 0
    kinds = {}
    for _ in range(max_moves):
        if max_evaluations is not None and evaluations >= max_evaluations:
            break
        candidates = []
        if "intra" in active_moves:
            move, count = _best_intra(routes, D)
            evaluations += count
            if move is not None:
                candidates.append(("intra", move[1][2], move))
        if "relocate" in active_moves:
            move, count = find_relocate_move(routes, D, demands, capacity)
            evaluations += count
            if move is not None:
                candidates.append(("relocate", move[4], move))
        if "swap" in active_moves:
            move, count = find_swap_move(routes, D, demands, capacity)
            evaluations += count
            if move is not None:
                candidates.append(("swap", move[4], move))
        if "2opt_star" in active_moves:
            move, count = find_two_opt_star_move(routes, D, demands, capacity)
            evaluations += count
            if move is not None:
                candidates.append(("2opt_star", move[4], move))
        if "cross" in active_moves:
            move, count = find_cross_exchange_move(routes, D, demands, capacity, max_segment)
            evaluations += count
            if move is not None:
                candidates.append(("cross", move[6], move))
        if not candidates:
            break
        kind, delta, move = min(candidates, key=lambda c: c[1])
        if kind == "intra":
            v, inner_move = move
            routes[v] = T.apply_two_opt_move(routes[v], inner_move)
        elif kind == "relocate":
            routes = apply_relocate_move(routes, move)
        elif kind == "swap":
            routes = apply_swap_move(routes, move)
        elif kind == "2opt_star":
            routes = apply_two_opt_star_move(routes, move)
        else:
            routes = apply_cross_exchange_move(routes, move)
        n_moves += 1
        kinds[kind] = kinds.get(kind, 0) + 1
        cost += delta
        if keep_steps:
            steps.append(Step((kind, move), [r.copy() for r in routes], cost))
    cost = T.solution_cost(routes, D)                                        # Rundungsfehler der Delta-Summe beseitigen
    return Descent(routes, cost, steps, evaluations, n_moves, kinds)
