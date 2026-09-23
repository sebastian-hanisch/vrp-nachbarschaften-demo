"""Auswertung der VRP-Nachbarschaften-Demo: Clarke-&-Wright-Konstruktion vs. + Intra-Route-Suche vs. + volle
Inter-Route-Nachbarschaft (Relocate/Swap/2-opt*/CROSS-exchange), bei gleichem Bewertungsbudget.

Anders als jedes bisherige Stück dieser Linie ist diese Suche vollständig DETERMINISTISCH: die Savings-
Konstruktion und die bestes-Verbesserung-Suche enthalten keinerlei Zufall - kein Ketten-Seed-Regler, keine
Streuung über Ketten (siehe README "Was nicht funktioniert hat"). Ein Vorschlag ist ein bewertetes (Zug, Kandidat)-
Paar - dieselbe Einheit über alle Zugarten hinweg, ein Tausch-/Verschiebe-Versuch, der Kapazität verletzt, wird
trotzdem als bewertet gezählt (er wird nur nicht ausgeführt).

Kennzahl: prozentuale Verbesserung gegenüber der Savings-Konstruktion (keine Held-Karp-artige untere Schranke -
eine straffe CVRP-Schranke ist ungleich aufwändiger als bei einer einzelnen TSP-Tour, siehe Grenzen)."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import vrpn_constants as C
import vrpn_construction as CO
import vrpn_interroute as IR
import vrpn_scenario as S
import vrpn_tour as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    capacity: float = C.DEFAULT_CAPACITY
    active_moves: tuple = IR.MOVE_TYPES
    budget: int = C.DEFAULT_BUDGET


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed, capacity):
    inst = S.generate(n, cluster_share, seed, capacity=capacity)
    return inst, T.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def construction(n, cluster_share, seed, capacity):
    inst, D = instance(n, cluster_share, seed, capacity)
    routes = CO.savings_construction(n, D, inst.demands, capacity)
    return tuple(tuple(r.tolist()) for r in routes)


def _routes_array(routes_tuple):
    return [np.array(r, dtype=np.int64) for r in routes_tuple]


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    construction_routes: list
    construction_cost: float
    run: object                     # IR.Descent
    n_routes: int

    @property
    def gap_of(self):
        def _f(cost):
            return 100.0 * (self.construction_cost - cost) / self.construction_cost
        return _f

    @property
    def improvement(self):
        """Prozentuale Verbesserung gegenüber der Savings-Konstruktion (positiv = besser)."""
        return self.gap_of(self.run.cost)


def analyse(settings):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed, settings.capacity)
    routes_tuple = construction(settings.n, settings.cluster_share, settings.seed, settings.capacity)
    routes = _routes_array(routes_tuple)
    construction_cost = T.solution_cost(routes, D)
    run = IR.descend(D, routes, inst.demands, settings.capacity, active_moves=settings.active_moves, keep_steps=False, max_evaluations=settings.budget)
    return Analysis(settings, inst, D, routes, construction_cost, run, len(run.routes))


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    """Mittel über die festen Instanzen für `base` mit `changes` (deterministisch je Instanz - kein Ketten-Mittel
    nötig, siehe Modul-Docstring)."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        a = analyse(replace(s0, seed=seed))
        rows.append({"improvement": a.improvement, "cost": a.run.cost, "construction_cost": a.construction_cost,
                     "n_routes": a.n_routes, "evaluations": a.run.evaluations, "n_moves": a.run.n_moves})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"improvement_sd": float(np.std([r["improvement"] for r in rows])), "n_runs": len(rows)})
    return out


SWEEP_VALUES = {"budget": C.BUDGETS, "capacity": (15, 30, 60, 120, 250, 600), "n": (10, 20, 40, 60, 100, 150, 200),
                "cluster_share": (0, 25, 50, 75, 100)}
SWEEP_LABELS = {"budget": "Budget (bewertete Kandidaten)", "capacity": "Kapazität", "n": "Stopps", "cluster_share": "Anteil in Gruppen (%)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def compare_active_moves(base_moves_configs, base=Settings()):
    """`base_moves_configs`: {Bezeichnung: active_moves-Tupel}. Gibt {Bezeichnung: run_config(...)} zurück - die
    Ablations-Messreihe (Konstruktion allein vs. +Intra vs. je einzelner Inter-Route-Typ vs. alle zusammen)."""
    return {label: run_config(base, active_moves=moves) for label, moves in base_moves_configs.items()}


SCALING_POLICIES = (("Budget 200 Tausend", lambda n: 200000), ("Budget 5 000 · Stopps", lambda n: 5000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in C.SCALING_N]} for label, fn in SCALING_POLICIES]
