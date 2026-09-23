"""Konstanten der VRP-Nachbarschaften-Demo: Szenario (geometrische Basis wortgleich zur Hill-Climbing-Demo, dazu
Bedarfe/Kapazität), Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
DEFAULT_BUDGET = 200000
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20

# --- VRP-eigen: Bedarfe und Kapazität -------------------------------------------------------------------------------------------------------
DEMAND_MIN, DEMAND_MAX = 1, 9      # rng.integers(DEMAND_MIN, DEMAND_MAX + 1) je Kunde
CAPACITY_MIN, CAPACITY_MAX, DEFAULT_CAPACITY, CAPACITY_STEP = 15, 600, 60, 5
# Kapazität >= Summe aller Bedarfe (bei n=60, mittlerer Bedarf 5: ~300, Worst Case 60*9=540) ergibt praktisch
# immer genau eine Route - CAPACITY_MAX=600 ist der "unendliche Kapazität"-Sonderfall dieses Reglers, der die
# Instanz auf die bestehende TSP-Wurzel reduziert (siehe tests/test_construction.py, tests/test_tour.py).
MAX_SEGMENT = 3                    # Segmentlänge für CROSS-exchange, wie das Or-opt der Wurzel

# --- Gemessene Werte (Mittel über 5 feste Sweep-Instanzen, Seeds 100000-100004; n=60, Kapazität 60, Budget 200
# --- Tausend, sofern nicht anders angegeben; 2026-09-23, alle Werte über ev.sweep/ev.run_config/ev.scaling_table/
# --- ev.compare_active_moves nachgerechnet, s. tests/test_claims.py) ---------------------------------------------
# ABLATION (Konstruktion vs. +Intra vs. +einzelner Inter-Route-Typ vs. volle Nachbarschaft), Kapazität 120 (dort
#   trägt Inter-Route am meisten bei, siehe Kapazitäts-Sweep unten): Konstruktion 0.00 %, +Intra 0.87 %, +Relocate
#   1.02 %, +Swap 0.99 %, +2-opt* 2.36 %, +CROSS-exchange 1.46 %, volle Nachbarschaft 2.47 % - 2-opt* trägt
#   ALLEIN fast so viel bei wie alle vier Inter-Route-Typen zusammen; CROSS-exchange bleibt trotz der bewiesenen
#   Verallgemeinerung (siehe tests/test_interroute.py) ALLEIN schwächer als 2-opt* allein, weil die Segmentlänge
#   hier auf MAX_SEGMENT=3 begrenzt ist (2-opt* kann beliebig lange Routen-Enden tauschen, das begrenzte
#   CROSS-exchange nicht).
# KAPAZITÄTS-SWEEP (Intra allein vs. volle Nachbarschaft), Kapazität 15/30/60/120/250/600 (Routenzahl im Mittel
#   23.8/11.4/6.0/3.0/2.0/1.0): Verbesserung ggü. Konstruktion 0.36/1.23/0.98/2.47/3.68/1.16 % - NICHT monoton mit
#   sinkender Kapazität (die Vorab-Vermutung "mehr Routen -> mehr Inter-Route-Wert" stimmt NICHT ungeprüft): der
#   Wert der Inter-Route-Nachbarschaft hat ein Optimum bei MITTLERER Kapazität (wenige, aber substanzielle
#   Routen) - bei sehr kleiner Kapazität sind die Routen zu winzig für sinnvolle Umbauten, bei sehr großer
#   Kapazität gibt es (ab einer Route) gar keine andere Route mehr zum Tauschen.
# BUDGET-SWEEP (Kapazität 60): 10T/25T/50T/100T/200T-2M: Verbesserung 0.49/0.84/0.91/0.98/0.98 % - konvergiert bei
#   ~100 Tausend Bewertungen (66 Tausend im Mittel gebraucht), danach kein weiterer Effekt.
# SKALIERUNG (Kapazität 60 fest), 20/40/60/100/150/200 Stopps, Budget 200T: Verbesserung 1.19/2.11/0.98/0.98/
#   0.57/0.20 % - sinkt bei großen Instanzen (mehr Routen, das feste Budget reicht relativ zur wachsenden
#   Routenzahl immer schlechter); mit wachsendem Budget (5000 · Stopps): 1.19/2.11/0.98/1.17/1.18/0.68 % - der
#   Rückgang schwächt sich ab, bleibt aber bei sehr großen Instanzen bestehen.
# TSP-SONDERFALL: bei Kapazität >= Gesamtbedarf (eine Route) erreicht die volle Pipeline (Konstruktion + Suche)
#   NACHWEISLICH exakt dieselbe Tour wie ein 2-opt-Abstieg der TSP-Wurzel von derselben Starttour aus (Differenz
#   0.0 über 5 Testinstanzen, siehe tests/test_interroute.py) - NICHT identisch mit der Wurzel-Demo eigenem
#   "2opt+oropt" (dort ist Or-opt Teil der Nachbarschaft, hier bewusst nicht, siehe Grenzen).


def _preset(capacity=DEFAULT_CAPACITY, active_moves=("intra", "relocate", "swap", "2opt_star", "cross"), budget=DEFAULT_BUDGET, n=DEFAULT_N):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "capacity": capacity, "active_moves": list(active_moves), "budget": budget}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Nur Intra-Route (Kontrolle)": _preset(active_moves=("intra",)),
    "Kleine Kapazität (viele Routen)": _preset(capacity=15),
    "Mittlere Kapazität (Inter-Route-Optimum)": _preset(capacity=120),
    "Sehr große Kapazität (TSP-Sonderfall)": _preset(capacity=CAPACITY_MAX),
    "Großes Budget (1 Million)": _preset(budget=1000000),
}
# Bei Instanz-Seed 35 (Standardfall); "Verbesserung" = Prozent ggü. der Savings-Konstruktion
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, Kapazität 60 (im Mittel 6 Routen), 200 Tausend Vorschläge: die volle Nachbarschaft verbessert die Savings-Konstruktion im Mittel um 0.98 % - reine Intra-Route-Suche allein nur um 0.61 %.",
    "Nur Intra-Route (Kontrolle)": "Dieselbe Instanz, aber OHNE Inter-Route-Züge (wie auf einer einzelnen TSP-Tour): 0.61 % statt 0.98 % - Relocate/Swap/2-opt*/CROSS-exchange bringen zusammen fast so viel wie die reine Intra-Route-Suche allein.",
    "Kleine Kapazität (viele Routen)": "Kapazität 15 (im Mittel knapp 24 winzige Routen): nur 0.36 % Verbesserung - die Routen sind zu klein für nennenswerte Umbauten, obwohl es hier am meisten potenzielle Tauschpartner gibt.",
    "Mittlere Kapazität (Inter-Route-Optimum)": "Kapazität 120 (im Mittel 3 Routen): 2.47 % Verbesserung - das gemessene Optimum des Kapazitäts-Reglers, größer als bei kleiner UND bei sehr großer Kapazität.",
    "Sehr große Kapazität (TSP-Sonderfall)": "Kapazität weit über dem Gesamtbedarf: genau EINE Route - die Instanz reduziert sich auf die TSP-Wurzel dieser Linie, nachweislich mit identischem Ergebnis zu deren 2-opt-Abstieg.",
    "Großes Budget (1 Million)": "1 Million statt 200 Tausend Vorschläge bei Standard-Kapazität: kein Unterschied (0.98 % beide) - die Suche konvergiert hier bereits deutlich unter 100 Tausend Bewertungen.",
}
# Beobachtete Spannweite der Verbesserung ggü. Konstruktion über die 5 festen Sweep-Instanzen (mit Sicherheitsabstand) -
# diese Suche ist deterministisch (kein Ketten-Seed), die Spannweite kommt allein aus der Instanz-Geometrie.
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": (0.0, 4.0),
    "Nur Intra-Route (Kontrolle)": (0.0, 4.0),
    "Kleine Kapazität (viele Routen)": (0.0, 2.0),
    "Mittlere Kapazität (Inter-Route-Optimum)": (0.5, 8.0),
    "Sehr große Kapazität (TSP-Sonderfall)": (0.0, 5.0),
    "Großes Budget (1 Million)": (0.0, 4.0),
}
