"""VRP-Nachbarschaften - Züge zwischen Routen, die auf einer einzelnen TSP-Tour gar nicht existieren können - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der "Konzepte"-Reihe der Trajektorien-Metaheuristiken-Linie, 3. Zweig des Nachbarschafts-Zweigs
(neben Lin-Kernighan und Dynasearch) - und der Aufbau der CVRP-Infrastruktur, die auch das letzte Stück dieser
Linie (ALNS) braucht. EIN gemeinsamer Instanz-Generator mit einem Kapazitäts-Regler: unendliche Kapazität ist
GENAU die TSP-Instanz der Wurzel dieser Linie (nachweislich, nicht nur strukturell - siehe README). Sobald mehrere
Fahrzeuge mit Kapazitätsgrenze mehrere Routen fahren, werden Inter-Route-Züge möglich (Relocate, Swap, 2-opt*,
CROSS-exchange), die es auf einer einzelnen Tour gar nicht geben kann. Bringen sie etwas gegenüber einer reinen
Savings-Konstruktion mit Intra-Route-Suche? Muss gemessen werden, nicht angenommen.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import numpy as np
import streamlit as st

import vrpn_constants as C
import vrpn_evaluation as EV
import vrpn_interroute as IR
import vrpn_tour as T
from vrpn_evaluation import SWEEP_LABELS, Settings, analyse, compare_active_moves, scaling_table, sweep
from vrpn_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from vrpn_visualization import build_ablation_bar, build_instance, build_routes, build_scaling, build_sweep

st.set_page_config(page_title="VRP-Nachbarschaften – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


@st.cache_data(show_spinner=False)
def _ablation(base):
    configs = {
        "Nur Konstruktion": (),
        "+ Intra-Route": ("intra",),
        "+ Relocate": ("intra", "relocate"),
        "+ Swap": ("intra", "swap"),
        "+ 2-opt*": ("intra", "2opt_star"),
        "+ CROSS-exchange": ("intra", "cross"),
        "Volle Nachbarschaft": IR.MOVE_TYPES,
    }
    return compare_active_moves(configs, base)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("🚚 VRP-Nachbarschaften – Züge zwischen Routen")
st.markdown(
    """
Auf einer einzelnen TSP-Tour wirken 2-opt und Or-opt nur INNERHALB dieser einen Route. Sobald mehrere Fahrzeuge
mit einer **Kapazitätsgrenze** mehrere Routen fahren, werden **Inter-Route-Züge** möglich, die es auf einer
einzelnen Tour gar nicht geben kann: **Relocate** (ein Kunde wandert in eine andere Route), **Swap** (zwei Kunden
tauschen die Route), **2-opt\\*** (die Enden zweier Routen werden vertauscht) und **CROSS-exchange** (ein Segment
wird zwischen zwei Routen getauscht - die Verallgemeinerung der drei anderen). Bringen diese Züge gegenüber einer
reinen Savings-Konstruktion mit Intra-Route-Suche etwas? Und wächst ihr Wert, wenn die Kapazität sinkt und mehr
Routen erzwungen werden?
"""
)
st.caption(
    "Zehntes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe, 3. Zweig des "
    "Nachbarschafts-Zweigs (neben [lin-kernighan-demo](https://github.com/sebastian-hanisch/lin-kernighan-demo) und "
    "[dynasearch-demo](https://github.com/sebastian-hanisch/dynasearch-demo)) - dieselbe geometrische Basis wie die "
    "gesamte Trajektorien-Metaheuristiken-Linie (Depot in der Mitte, Kundenstopps in einem 100 × 100-km-Gebiet), "
    "um Bedarfe je Kunde und eine Fahrzeug-Kapazität erweitert. Kapazität >= Gesamtbedarf ist GENAU die "
    "TSP-Instanz der [hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) - nachweislich, "
    "nicht nur strukturell (siehe README)."
)

with st.expander("So funktionieren die Inter-Route-Züge", expanded=True):
    st.markdown(
        """
1. **Savings-Konstruktion** (Clarke & Wright): startet mit einer Route je Kunde, fusioniert Routenpaare in
   absteigender Ersparnis, solange Kapazität und Wirtschaftlichkeit es erlauben - EIN Konstruktionsverfahren,
   nicht selbst der Gegenstand der Messung.
2. **Intra-Route-2-opt**: wie bei der TSP-Wurzel, aber als offener Pfad (das Depot gehört zu MEHREREN Routen
   gleichzeitig und darf in keiner von ihnen frei "gedreht" werden).
3. **Relocate**: ein Kunde wandert von einer Route in eine ANDERE.
4. **Swap**: je ein Kunde aus zwei verschiedenen Routen tauscht die Position.
5. **2-opt\\*** (Potvin & Rousseau 1995): die Enden zweier Routen ab je einer Position werden vertauscht.
6. **CROSS-exchange** (Taillard et al. 1997): ein Segment (Länge 1-3) wird zwischen zwei Routen getauscht -
   beweisbar die Verallgemeinerung von Swap (Segmentlänge 1/1) und 2-opt\\* (Segmentlänge bis Routenende).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu).",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.",
    )
    capacity = st.slider(
        "Fahrzeug-Kapazität", *bounds("capacity_slider"), key="capacity_slider", step=C.CAPACITY_STEP,
        help="Sehr klein = viele winzige Routen, sehr groß = eine einzige Route (der TSP-Sonderfall). Der Wert der Inter-Route-Züge ist NICHT monoton: er hat ein Optimum bei MITTLERER Kapazität (2.47 % bei Kapazität 120), nicht bei sehr kleiner (0.36 % bei 15) oder sehr großer (0 % bei einer Route).",
    )
    moves = st.multiselect(
        "Aktive Zugarten", options=list(IR.MOVE_TYPES), key="moves_select",
        format_func=lambda m: IR.MOVE_LABELS[m],
        help="Für die Ablations-Messreihe: 2-opt* trägt bei mittlerer Kapazität fast so viel bei wie alle vier Inter-Route-Typen zusammen (2.36 % gegen 2.47 %), CROSS-exchange allein bleibt trotz der Verallgemeinerung schwächer (auf Segmentlänge 3 begrenzt).",
    )
    budget = st.select_slider(
        "Budget (bewertete Kandidaten)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Konvergiert bei 60 Stopps/Kapazität 60 im Mittel bereits unter 100 Tausend Bewertungen.",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für Lage und Bedarfe der Stopps.")
    st.caption("Kein Ketten-Seed-Regler: anders als jedes bisherige Stück dieser Linie ist diese Suche vollständig deterministisch (Savings-Konstruktion + bestes-Verbesserung-Suche, kein Zufall im Kern).")

if not moves:
    st.warning("Mindestens eine Zugart muss aktiv sein - Intra-Route wird automatisch wieder aktiviert.")
    moves = ["intra"]

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed),
    "capacity_slider": int(capacity), "moves_select": tuple(moves), "budget_select": int(budget),
})

active_moves = tuple(m for m in IR.MOVE_TYPES if m in moves)
settings = Settings(int(n_stops), int(cluster_share), int(seed), float(capacity), active_moves, int(budget))
with st.spinner("Rechne..."):
    a = _analysis(settings)
xy = a.inst.xy

# --- VRP-Nachbarschaften in Aktion ------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 VRP-Nachbarschaften in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Konstruktion", 3: "3 · Ergebnis"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="vrpn_step", format_func=lambda s: STEP_LABELS[s])

if step == 1:
    st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen, Gesamtbedarf {a.inst.total_demand:.0f}, Kapazität {settings.capacity:.0f}")
    st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
elif step == 2:
    st.markdown(f"**Savings-Konstruktion**: {len(a.construction_routes)} Routen, {a.construction_cost:.1f} km")
    st.plotly_chart(build_routes(xy, a.construction_routes), width="stretch", key="s2_map")
else:
    st.markdown(f"**Nach der Suche**: {a.n_routes} Routen, {a.run.cost:.1f} km – {a.improvement:.2f} % besser als die Konstruktion")
    st.plotly_chart(build_routes(xy, a.run.routes), width="stretch", key="s3_map")
    if a.run.kinds:
        st.caption("Angewandte Züge: " + ", ".join(f"{IR.MOVE_LABELS[k]} ×{v}" for k, v in a.run.kinds.items()))

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Suche gefunden hat")
st.caption(
    "**Verbesserung:** Prozent gegenüber der Savings-Konstruktion (keine untere Schranke wie bei der TSP-Wurzel - "
    "eine straffe CVRP-Schranke ist ungleich aufwändiger, siehe README). Diese Suche ist deterministisch: derselbe "
    "Lauf liefert immer dasselbe Ergebnis."
)
m1, m2, m3 = st.columns(3)
m1.metric("Verbesserung ggü. Konstruktion", f"{a.improvement:.2f} %", delta=f"{a.run.n_moves} Züge", delta_color="off")
m2.metric("Routen", f"{a.n_routes}", delta=f"aus {len(a.construction_routes)} nach Konstruktion", delta_color="off")
m3.metric("Länge (km)", f"{a.run.cost:.1f}", delta=f"Konstruktion {a.construction_cost:.1f}", delta_color="off")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    st.table({"": ["Länge (km)", "Routen", "Bewertete Kandidaten"],
              "Konstruktion": [f"{a.construction_cost:.1f}", f"{len(a.construction_routes)}", "-"],
              "Nach der Suche": [f"{a.run.cost:.1f}", f"{a.n_routes}", _fmt_int(a.run.evaluations)]})
with d2:
    st.markdown("**Routendemand**")
    loads = [T.route_demand(r, a.inst.demands) for r in a.run.routes]
    st.table({"Route": [f"{i + 1}" for i in range(len(loads))], "Bedarf": [f"{l:.0f}" for l in loads],
              "Kapazität": [f"{settings.capacity:.0f}" for _ in loads]})

st.markdown("---")

# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Kapazität und Budget ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 10 bis 60 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste.")

st.markdown("---")

# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Welche Zugart trägt am meisten bei?")
if st.button("Ablation über alle Zugarten berechnen (dauert etwa 20 Sekunden)", key="ablation_start"):
    st.session_state["ablation_on"] = True
if st.session_state.get("ablation_on"):
    with st.spinner("Rechne Konstruktion, +Intra, +je einen Inter-Route-Typ, volle Nachbarschaft über 5 Instanzen..."):
        rows_abl = _ablation(replace(base_sweep, capacity=120))
    labels = list(rows_abl.keys())
    st.plotly_chart(build_ablation_bar(labels, [rows_abl[l]["improvement"] for l in labels]), width="stretch", key="ablation_chart")
    st.caption("Mittel über 5 feste Instanzen bei Kapazität 120 (dort trägt Inter-Route am meisten bei, siehe Kapazitäts-Sweep). "
               "2-opt* trägt allein fast so viel bei wie alle vier Inter-Route-Typen zusammen - CROSS-exchange bleibt trotz der bewiesenen Verallgemeinerung schwächer, weil die Segmentlänge hier auf 3 begrenzt ist.")

st.markdown("---")

st.subheader("🔬 Skalierung: wie verändert sich die Verbesserung mit der Instanzgröße?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 60 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen (Einstellungen wie in der Seitenleiste außer Stopps und Budget, feste Kapazität). "
               "Bei festem 200-Tausend-Budget sinkt die Verbesserung bei großen Instanzen (mehr Routen, das Budget reicht relativ schlechter) - "
               "mit wachsendem Budget (5000 · Stopps) schwächt sich der Rückgang ab, bleibt aber bestehen.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Kapazität liegt in einem mittleren Bereich** | Der Wert der Inter-Route-Nachbarschaft ist NICHT monoton - bei sehr kleiner Kapazität (0.36 % bei Kapazität 15) sind die Routen zu winzig für Umbauten, bei sehr großer (0 % bei einer Route) gibt es keine andere Route mehr zum Tauschen. Das Optimum liegt dazwischen (2.47 % bei Kapazität 120). | (kein Nachfolger nötig - eine echte, gemessene Eigenschaft, keine Lücke) |
| **Kein Or-opt innerhalb einer Route** | Intra-Route-Suche ist hier bewusst auf 2-opt beschränkt (Or-opt wäre der Spezialfall des CROSS-exchange mit gleicher Quell- und Zielroute, kein eigenes Modul) - der TSP-Sonderfall reduziert deshalb auf einen reinen 2-opt-, nicht 2opt+oropt-Abstieg der Wurzel. | **ALNS** (destroy/repair-Operatoren, letztes Stück dieser Linie, baut auf dieser Infrastruktur auf) |
| **CROSS-exchange-Segmente werden nicht umgekehrt** | Nur die Reihenfolge wird getauscht, keine Spiegelung des Segments - eine bewusste Vereinfachung gegenüber der vollen Literatur-Definition, die auch Umkehrung erlaubt. | (kein Nachfolger nötig - eine dokumentierte Vereinfachung) |
| **Kein Zufall im Kern** | Anders als jedes bisherige Stück dieser Linie streut diese Suche NICHT über Ketten - Savings-Konstruktion und bestes-Verbesserung-Suche sind vollständig deterministisch. | (kein Nachfolger nötig - eine echte methodische Eigenschaft dieses Stücks) |
| **Keine untere Schranke** | Anders als die TSP-Wurzel (Held-Karp) gibt es hier keine berechnete untere Schranke - "Verbesserung" ist relativ zur Savings-Konstruktion, nicht zum Optimum. | (kein Nachfolger nötig - eine straffe CVRP-Schranke wäre ein eigenes, aufwändiges Thema) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** $N=n+1$ Knoten (Depot + $n$ Kunden mit Bedarf $q_i$), euklidische Entfernungen $d_{ij}$. Gesucht:
eine Menge von Routen (jede beginnt und endet am Depot), die jeden Kunden genau einmal bedienen, deren
Gesamtbedarf je Route die Kapazität $Q$ nicht übersteigt, bei minimaler Gesamtlänge.

**Relocate.** Kunde $c$ an Position $p$ in Route $A$ entfernen (Gewinn $d_{\text{prev},c}+d_{c,\text{next}}-
d_{\text{prev},\text{next}}$), an Position $q$ in Route $B$ einfügen (Kosten analog) - Δ = Einfügekosten - Gewinn.

**Swap(1,1).** Kunden $c_1 \in A$, $c_2 \in B$ tauschen die Position - Δ ergibt sich aus den vier geänderten
Kanten um $c_1$ und $c_2$.

**2-opt\\* (Potvin & Rousseau 1995).** $A' = A[:i] + B[j:]$, $B' = B[:j] + A[i:]$ - zwei entfernte, zwei neue Kante.

**CROSS-exchange (Taillard et al. 1997).** Segmente $A[p_1{:}p_1{+}l_1]$ und $B[p_2{:}p_2{+}l_2]$ tauschen die
Route (feste Reihenfolge). Bei $l_1=l_2=1$ **exakt** Swap, bei Segmenten bis zum Routenende **exakt** 2-opt\\*
(beide Spezialfälle regressionsgeprüft, siehe README Verifikation).

**Kennzahl.** Verbesserung $= 100 \cdot (L_{\text{Konstruktion}} - L)/L_{\text{Konstruktion}}$.

**Literatur.** Potvin, J.-Y., & Rousseau, J.-M. (1995). *An Exchange Heuristic for Routeing Problems with Time
Windows.* Journal of the Operational Research Society, 46(12), 1433-1446. Taillard, E. D., Badeau, P., Gendreau,
M., Guertin, F., & Potvin, J.-Y. (1997). *A Tabu Search Heuristic for the Vehicle Routing Problem with Time
Windows.* Transportation Science, 31(2), 170-186. Clarke, G., & Wright, J. W. (1964). *Scheduling of Vehicles
from a Central Depot to a Number of Delivery Points.* Operations Research, 12(4), 568-581.

Implementiert in `vrpn_tour.py` (Routendarstellung, Intra-Route-2-opt), `vrpn_construction.py`
(Clarke-&-Wright-Savings), `vrpn_interroute.py` (die vier Inter-Route-Züge + kombinierte Suche),
`vrpn_scenario.py` (Instanzen), `vrpn_evaluation.py` (Kennzahlen, Sweeps, Ablation, Skalierung).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
