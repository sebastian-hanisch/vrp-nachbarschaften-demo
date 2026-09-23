"""Clarke & Wright Savings-Konstruktion (klassisch, symmetrische Entfernungen) - EIN Konstruktionsverfahren, bewusst
NICHT der Gegenstand der Messung dieses Stücks (wie jedes Stück dieser Linie: ein Baustein wird als
Vergleichsgröße fixiert, nicht selbst erforscht). Startet mit einer Route je Kunde, fusioniert Routenpaare in
absteigender Ersparnis-Reihenfolge, solange die Fusion tatsächlich Strecke spart UND die Kapazität einhält.

Kein fester Fahrzeug-Anzahl-Parameter (anders als manche Lehrbuch-Varianten): die Zahl der Routen ergibt sich
allein aus Kapazität und Geometrie. Bei sehr großer Kapazität bleiben nur noch Fusionen mit negativer Ersparnis
übrig (die klassische Abbruchbedingung) - ob das Ergebnis dann bereits eine einzige Route ist, hängt von der
Instanz ab; die Behauptung "Kapazität >= Gesamtbedarf reduziert auf die TSP-Wurzel" gilt für die VOLLE Pipeline
(Konstruktion + lokale Suche mit Inter-Route-Zügen), nicht für die Konstruktion allein - siehe
tests/test_construction.py und tests/test_interroute.py."""

import numpy as np


def savings_construction(n, D, demands, capacity):
    """`n` Kunden (Indizes 1..n), `D` Distanzmatrix (n+1 x n+1, Index 0 = Depot). Gibt eine Liste von
    `np.ndarray`-Routen zurück (Kundenindizes, ohne Depot)."""
    routes = {i: [i] for i in range(1, n + 1)}
    route_of = {i: i for i in range(1, n + 1)}
    loads = {i: float(demands[i]) for i in range(1, n + 1)}

    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = D[i, 0] + D[0, j] - D[i, j]
            if s > 0:
                savings.append((s, i, j))
    savings.sort(key=lambda t: -t[0])

    for _s, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        route_i, route_j = routes[ri], routes[rj]
        if loads[ri] + loads[rj] > capacity:
            continue
        merged = _try_merge(route_i, route_j, i, j)
        if merged is None:
            continue
        for c in route_j:
            route_of[c] = ri
        routes[ri] = merged
        loads[ri] += loads[rj]
        del routes[rj]
        del loads[rj]

    return [np.array(r, dtype=np.int64) for r in routes.values()]


def _try_merge(route_i, route_j, i, j):
    """Verbindet zwei Routen an den Enden i (in route_i) und j (in route_j), sofern i UND j tatsächlich an einem
    Ende ihrer jeweiligen Route liegen (klassische Clarke-&-Wright-Bedingung - innere Kunden können eine Route
    nicht verlängern). Richtet beide Routen bei Bedarf um (symmetrische Entfernungen: eine Route rückwärts zu
    fahren kostet dasselbe)."""
    if route_i[-1] == i:
        left = route_i
    elif route_i[0] == i:
        left = route_i[::-1]
    else:
        return None
    if route_j[0] == j:
        right = route_j
    elif route_j[-1] == j:
        right = route_j[::-1]
    else:
        return None
    return left + right
