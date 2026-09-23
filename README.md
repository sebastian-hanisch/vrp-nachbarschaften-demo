# VRP-Nachbarschaften – Züge zwischen Routen – Streamlit-Demo

Zehntes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", 3. Zweig des **Nachbarschafts-Zweigs** (neben [lin-kernighan-demo](../lin-kernighan-demo) und [dynasearch-demo](../dynasearch-demo)) - und der Aufbau der **CVRP-Infrastruktur**, die auch das letzte Stück dieser Linie (ALNS) braucht:
dieselbe geometrische Basis wie [hill-climbing-demo](../hill-climbing-demo) und alle Geschwisterstücke (Depot in der Mitte, Kundenstopps in einem 100 × 100-km-Gebiet), um Bedarfe je Kunde und eine Fahrzeug-Kapazität erweitert.

**Einordnung in die Reihe:** Auf einer einzelnen TSP-Tour gibt es nur EINE Route - 2-opt und Or-opt wirken nur INNERHALB dieser einen Route. Sobald mehrere Fahrzeuge mit Kapazitätsgrenze mehrere Routen fahren, werden **Inter-Route-Züge** möglich, die auf einer einzelnen Tour gar nicht existieren können: **Relocate**, **Swap**, **2-opt\*** (Potvin & Rousseau 1995) und **CROSS-exchange** (Taillard et al. 1997). EIN gemeinsamer Instanz-Generator mit einem Kapazitäts-Regler: **Kapazität ≥ Gesamtbedarf ist GENAU die TSP-Instanz der Wurzel** - nicht nur strukturell, sondern nachweislich (die volle Pipeline erreicht bei einer Route exakt dasselbe Ergebnis wie ein 2-opt-Abstieg der TSP-Wurzel von derselben Starttour, siehe Verifikation).
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo ── parallel-tempering-demo                        [gebaut]
  ├─ iterated-local-search-demo ── variable-neighborhood-search-demo            [gebaut]
  ├─ tabu-search-demo                                                            [gebaut]
  ├─ grasp-demo                                                                  [gebaut]
  └─ Nachbarschafts-Zweig
        ├─ lin-kernighan-demo (variable Tiefe statt fixer 2-opt-Nachbarschaft)  [gebaut]
        ├─ dynasearch-demo (viele unabhängige Züge auf einmal statt einer)      [gebaut]
        └─ vrp-nachbarschaften-demo (Züge ZWISCHEN Routen, CVRP-Infrastruktur)  [dieses Stück]
              └─ ALNS (destroy/repair, baut auf dieser Infrastruktur auf)        [nicht gebaut]
```

Ergebnis in Kürze: **Inter-Route-Züge helfen klar** - bei mittlerer Kapazität verdoppelt sich die Verbesserung gegenüber reiner Intra-Route-Suche fast (2.47 % gegen 0.87 % bei Kapazität 120). Aber die Vorab-Vermutung "mehr Routen (kleinere Kapazität) heißt mehr Wert für Inter-Route-Züge" stimmt **nicht**: der Wert hat ein **Optimum bei mittlerer Kapazität**, nicht bei kleiner (0.36 % bei Kapazität 15, die Routen sind zu winzig für sinnvolle Umbauten) oder sehr großer (0 % bei einer Route, es gibt keine andere Route mehr zum Tauschen). Unter den vier Zugarten trägt **2-opt\*** allein fast so viel bei wie alle vier zusammen (2.36 % gegen 2.47 %) - **CROSS-exchange**, obwohl es 2-opt\* beweisbar verallgemeinert, bleibt allein schwächer (1.46 %), weil die Segmentlänge hier auf 3 begrenzt ist.

| Frage | Ergebnis (60 gleichverteilte Stopps, Kapazität 60, 200 Tausend Vorschläge, sofern nicht anders angegeben; Mittel über 5 feste Instanzen, Seeds 100000–100004; Verbesserung = Prozent ggü. der Savings-Konstruktion) |
|---|---|
| Standardfall | ✅ volle Nachbarschaft **0.98 %** besser als die Konstruktion - reine Intra-Route-Suche allein nur **0.61 %** |
| **Ablation (bei Kapazität 120, dort am wirksamsten)** | ⚠️ Konstruktion 0.00 %, +Intra 0.87 %, +Relocate 1.02 %, +Swap 0.99 %, **+2-opt\* 2.36 %**, +CROSS-exchange 1.46 %, volle Nachbarschaft **2.47 %** |
| **Kapazitäts-Sweep** | ⚠️ 15/30/60/120/250/600: **0.36**/1.23/0.98/**2.47**/**3.68**/1.16 % - NICHT monoton, Optimum bei mittlerer Kapazität |
| **Budget-Sweep** | ✅ 10T/25T/50T/100T-2M: 0.49/0.84/0.91/**0.98 %** (konvergiert, ~66 Tausend Bewertungen im Mittel gebraucht) |
| **Skalierung (200-Tausend-Budget, Kapazität 60 fest)** | ⚠️ 20/40/60/100/150/200 Stopps: 1.19/2.11/0.98/0.98/0.57/**0.20 %** - sinkt bei großen Instanzen |
| **TSP-Sonderfall (Kapazität ≥ Gesamtbedarf)** | ✅ genau EINE Route, Ergebnis bytegleich zum 2-opt-Abstieg der TSP-Wurzel von derselben Starttour (5/5 Testinstanzen) |

## Was die Demo zeigt

1. **VRP-Nachbarschaften in Aktion** (Schritt-Slider): **Instanz** → **Konstruktion** (Savings-Routen, mehrfarbig) → **Ergebnis** (Routen nach der Suche, angewandte Züge je Art).
2. **Was die Suche gefunden hat:** Verbesserung ggü. Konstruktion, Routenzahl, Länge, Routendemand je Route.
3. **📐 Sweeps** über Kapazität, Budget, Stopps und Gruppen (5 feste Instanzen ab Seed 100000).
4. **🔬 Experimente auf Abruf:** Ablation über alle fünf Zugarten (Konstruktion → +Intra → +je ein Inter-Route-Typ → volle Nachbarschaft); Skalierung von 20 bis 200 Stopps bei festem und wachsendem Budget.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Kapazitäts-Optimum, kein intra-route Or-opt, keine CROSS-exchange-Segmentumkehrung, kein Zufall im Kern, keine untere Schranke).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **Fahrzeug-Kapazität** (15–600, der Kern-Regler dieses Stücks), **aktive Zugarten** (Mehrfachauswahl für die Ablations-Messreihe), Budget (10 Tausend bis 2 Millionen bewertete Kandidaten), Seed der Instanz (+ 🎲). **Kein Ketten-Seed-Regler**: anders als jedes bisherige Stück dieser Linie ist diese Suche vollständig deterministisch (Savings-Konstruktion + bestes-Verbesserung-Suche, kein Zufall im Kern) - siehe Grenzen.

## Messwerte der Presets

| Preset | Verbesserung (Instanz-Seed 35) |
|---|---|
| Standardfall (Voreinstellung) | 0.26 % |
| Nur Intra-Route (Kontrolle) | 0.17 % |
| Kleine Kapazität (viele Routen) | 0.44 % |
| Mittlere Kapazität (Inter-Route-Optimum) | 6.49 % |
| Sehr große Kapazität (TSP-Sonderfall) | 1.15 %, genau eine Route |
| Großes Budget (1 Million) | 0.26 % (kein Unterschied zum Standardfall - konvergiert längst) |

Die einzelne Standardinstanz (Seed 35) zeigt dasselbe Muster wie die Sweep-Mittelwerte, aber deutlicher (6.49 % statt 2.47 % bei mittlerer Kapazität) - die Mittelwerte oben in der Ergebnis-Tabelle sind die belastbaren Zahlen; jedes Preset prüft sich zusätzlich über die 5 festen Sweep-Instanzen gegen eine gemessene Spannweite (siehe `tests/test_presets.py`).

## Modell und Verfahren

- **Instanz** (`vrpn_scenario.py`): geometrische Basis (Depot, Kundenverteilung/Gruppierung) wortgleich zu `hc_scenario.generate` der Hill-Climbing-Demo (bytegleich kreuzgeprüft, siehe Verifikation), um Bedarfe je Kunde und eine Kapazität erweitert.
- **Routendarstellung + Intra-Route-2-opt** (`vrpn_tour.py`): eine Route ist ein offener Pfad, Depot an beiden Enden fest verankert (es gehört zu mehreren Routen gleichzeitig und darf in keiner von ihnen frei "gedreht" werden) - EIGENSTÄNDIG hergeleitet, nicht einfach die zyklische Formel der TSP-Wurzel wiederverwendet, aber nachweislich (nicht nur vermutet) gleichwertig im Sonderfall einer einzelnen Route (siehe Verifikation).
- **Savings-Konstruktion** (`vrpn_construction.py`): Clarke & Wright (1964), klassisch - EIN Konstruktionsverfahren, bewusst nicht selbst der Gegenstand der Messung.
- **Inter-Route-Züge** (`vrpn_interroute.py`): Relocate, Swap, 2-opt\* (Potvin & Rousseau 1995), CROSS-exchange (Taillard et al. 1997) - jede Delta-Formel gegen direkte Kostenneuberechnung kreuzgeprüft, die behauptete Verallgemeinerung CROSS-exchange ⊇ {Swap, 2-opt\*} bewiesen, nicht nur behauptet.
- **Kombinierte Suche** (`vrpn_interroute.descend`): bestes-Verbesserung über die kombinierte Kandidatenmenge aller aktivierten Zugarten, voller Rescan je Iteration - "ein Vorschlag = ein bewertetes (Zug, Kandidat)-Paar", dieselbe Konvention wie jedes Stück dieser Linie.
- **Auswertung** (`vrpn_evaluation.py`): Kennzahlen, Sweeps, Ablation, Skalierung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "mehr Routen (kleinere Kapazität) heißt mehr Wert für Inter-Route-Züge"** – **widerlegt**. Der Wert hat ein Optimum bei MITTLERER Kapazität (2.47 % bei Kapazität 120, im Mittel 3 Routen) - bei sehr kleiner Kapazität (15, ~24 winzige Routen) sind die Routen zu klein für sinnvolle Umbauten (nur 0.36 %), bei sehr großer (eine Route) gibt es strukturell keine andere Route mehr zum Tauschen (0 % Zusatz). Ein weiteres nicht-monotones Optimum in dieser Linie - inzwischen fast die Regel, nicht die Ausnahme.
- **2-opt\* trägt allein fast so viel bei wie die volle Nachbarschaft** (2.36 % gegen 2.47 % bei Kapazität 120) - **CROSS-exchange bleibt allein schwächer** (1.46 %), obwohl es 2-opt\* beweisbar verallgemeinert: die Segmentlänge ist hier auf `MAX_SEGMENT=3` begrenzt (wie das Or-opt der Wurzel), 2-opt\* kann dagegen beliebig lange Routen-Enden tauschen. Ein ehrlicher, erklärbarer Befund - keine Implementierungslücke, sondern eine bewusste Tractability-Grenze.
- **Kein Intra-Route-Or-opt**: die Intra-Route-Suche ist hier auf 2-opt beschränkt (Or-opt wäre der Spezialfall des CROSS-exchange mit gleicher Quell- und Zielroute - kein eigenes Modul). Der TSP-Sonderfall reduziert deshalb auf einen reinen 2-opt-, nicht 2opt+oropt-Abstieg der Wurzel (bytegleich geprüft gegen genau diesen, nicht gegen den stärkeren).
- **CROSS-exchange-Segmente werden nicht umgekehrt** (nur die Reihenfolge wird getauscht) - eine bewusste Vereinfachung gegenüber der vollen Literatur-Definition.
- **Kein Zufall im Kern**: anders als jedes bisherige Stück dieser Linie streut diese Suche nicht über Ketten - Savings-Konstruktion und bestes-Verbesserung-Suche sind vollständig deterministisch. Kein Ketten-Seed-Regler, keine Streuungs-Experimente.
- **Keine untere Schranke**: anders als die TSP-Wurzel (Held-Karp) gibt es hier keine berechnete untere Schranke - "Verbesserung" ist relativ zur Savings-Konstruktion, nicht zum Optimum. Eine straffe CVRP-Schranke ist ungleich aufwändiger als die 1-Baum-Schranke einer einzelnen TSP-Tour.
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, Bedarfe unabhängig gleichverteilt (1-9), keine Zeitfenster. Zeiten hängen vom Rechner ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Jede Delta-Formel der vier Inter-Route-Züge gegen direkte Kostenneuberechnung kreuzgeprüft** (vorher/nachher, nicht nur der Formel vertraut) - für Relocate, Swap, 2-opt\* und CROSS-exchange einzeln, über mehrere Zufallsinstanzen.
- **Die behauptete Verallgemeinerung CROSS-exchange ⊇ {Swap, 2-opt\*} tatsächlich bewiesen, nicht nur behauptet**: bei Segmentlänge 1/1 liefert CROSS-exchange an JEDER Position exakt dasselbe Delta wie die eigenständige Swap-Formel; bei Segmenten bis zum Routenende exakt dasselbe wie die eigenständige 2-opt\*-Formel (beide algebraisch hergeleitet UND an vielen Positionen numerisch geprüft, siehe `tests/test_interroute.py`).
- **Der TSP-Sonderfall nachgewiesen, nicht nur strukturell angenommen**: die Pfad-2-opt-Nachbarschaft einer einzelnen Route sieht auf den ersten Blick wie eine echte Teilmenge der zyklischen Nachbarschaft der TSP-Wurzel aus - tatsächlich erreicht sie nachweislich dieselben lokalen Optima (80 Zufallsinstanzen, n=10..80, Differenz in JEDEM Fall exakt 0 - jede zyklische 2-opt-Umkehrung hat eine redundante komplementäre Darstellung, die die Pfad-Bedingung nicht ausschließt). Die volle Pipeline (Konstruktion + Suche) kollabiert bei Kapazität ≥ Gesamtbedarf nachweislich auf eine einzige Route mit bytegleichem Ergebnis zum 2-opt-Abstieg der TSP-Wurzel von derselben Starttour (5 Testinstanzen).
- **Kapazitäts-Machbarkeit nie verletzt** (über viele Zufallsinstanzen geprüft), gültige Partition (jeder Kunde in genau einer Route), monotoner Abstieg der kombinierten Suche.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Ablations-, Kapazitäts-, Budget- und Skalierungs-Aussagen; positive **und** negative Aussagen), über dieselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`/`ev.compare_active_moves`), NIE über ein Ad-hoc-Skript mit abweichender Zufalls-Bindung (die Lehre aus der [lin-kernighan-demo](../lin-kernighan-demo) dieser Linie); alle 6 Presets über die 5 festen Sweep-Instanzen in gemessenen Spannweiten; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt, leere Zugarten-Auswahl, Würfel-Knopf, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).
- Übernommener Kern: 2-opt-Formel-Bausteine, Distanzmatrix (aus der Hill-Climbing-Demo); die Cross-Repo-Kreuzprüfungen (Geometrie und TSP-Sonderfall gegen `hill-climbing-demo`) laufen nur lokal (eigenständiges Repo, CI/frische Klone überspringen sie automatisch - wie der ortools-Kreuzvergleich der übrigen Linie).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Ablation, Skalierung), 🚧 Grenzen, Mathe |
| `vrpn_tour.py` | Routendarstellung, Kosten/Machbarkeit, Intra-Route-2-opt (offener Pfad) |
| `vrpn_construction.py` | Clarke-&-Wright-Savings-Konstruktion |
| `vrpn_interroute.py` | Relocate, Swap, 2-opt\*, CROSS-exchange + kombinierte Suche |
| `vrpn_scenario.py`, `vrpn_constants.py` | Instanzen (Geometrie + Bedarfe/Kapazität); Konstanten, Presets |
| `vrpn_evaluation.py` | Kennzahlen, Sweeps, Ablation, Skalierung |
| `vrpn_presets.py`, `vrpn_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt, mehrfarbige Routenkarte) |
| `tests/` | Übernommener Kern, Konstruktion, die zentrale Inter-Route-Korrektheitskette, Szenario/Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
