"""Presets: Vollständigkeit, gültige Werte, Verbesserung bleibt in der gemessenen Spannweite über die 5 festen
Sweep-Instanzen (kein Ketten-Mittel nötig - diese Suche ist deterministisch), Permalink-Konstanten."""

import pytest

import vrpn_constants as C
import vrpn_evaluation as ev
import vrpn_presets as P


def _settings(p, seed=None):
    return ev.Settings(n=p["n"], cluster_share=p["ballung"], seed=p["seed"] if seed is None else seed,
                        capacity=p["capacity"], active_moves=tuple(p["active_moves"]), budget=p["budget"])


def test_every_preset_has_help_bands_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS) and len(C.PRESETS) == 6
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0
        assert C.BALLUNG_MIN <= p["ballung"] <= C.BALLUNG_MAX and p["ballung"] % C.BALLUNG_STEP == 0
        assert p["budget"] in C.BUDGETS
        assert all(m in P.IR.MOVE_TYPES for m in p["active_moves"]) and len(p["active_moves"]) > 0
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(p[key] if key != "active_moves" else ",".join(p[key]))


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_improvement_stays_in_its_measured_band_over_instances(name):
    p = C.PRESETS[name]
    lo, hi = C.PRESET_EXPECTED_BANDS[name]
    for seed in C.SWEEP_SEEDS:
        a = ev.analyse(_settings(p, seed=seed))
        assert lo <= a.improvement <= hi, (seed, a.improvement)
    a_default = ev.analyse(_settings(p))
    assert lo <= a_default.improvement <= hi


def test_medium_capacity_preset_clearly_beats_small_and_very_large_capacity():
    """Die zentrale, nicht-monotone Kapazitäts-Erkenntnis: das gemessene Optimum liegt bei MITTLERER Kapazität,
    nicht bei kleiner oder sehr großer - hier direkt an den drei Presets geprüft."""
    small = ev.run_config(_settings(C.PRESETS["Kleine Kapazität (viele Routen)"]))
    medium = ev.run_config(_settings(C.PRESETS["Mittlere Kapazität (Inter-Route-Optimum)"]))
    large = ev.run_config(_settings(C.PRESETS["Sehr große Kapazität (TSP-Sonderfall)"]))
    assert medium["improvement"] > small["improvement"] + 0.5
    assert medium["improvement"] > large["improvement"] + 0.5


def test_no_swap_preset_disables_only_intra_route():
    assert C.PRESETS["Nur Intra-Route (Kontrolle)"]["active_moves"] == ["intra"]
    assert all(len(p["active_moves"]) == 5 for name, p in C.PRESETS.items() if name != "Nur Intra-Route (Kontrolle)")


def test_bounds_and_snapping_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert P.STEPS == {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.DEFAULT_BUDGET in C.BUDGETS


def test_moves_permalink_roundtrip():
    encoded = P._moves_to_str(("intra", "swap", "cross"))
    decoded = P._moves_from_str(encoded)
    assert decoded == ("intra", "swap", "cross")                                 # feste MOVE_TYPES-Reihenfolge
    with pytest.raises(ValueError):
        P._moves_from_str("")
