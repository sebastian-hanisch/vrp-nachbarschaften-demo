"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio, siehe bf_presets.py in bfs-demo)."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import vrpn_constants as C
import vrpn_interroute as IR


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _int_choice(options):
    def cast(value):
        value = int(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _moves_from_str(value):
    parts = [p for p in str(value).split(",") if p]
    moves = tuple(m for m in IR.MOVE_TYPES if m in parts)                    # feste Reihenfolge, unbekannte Werte ignoriert
    if not moves:
        raise ValueError(value)
    return moves


def _moves_to_str(moves):
    return ",".join(moves)


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "ballung_slider": SettingSpec("ballung", int, C.DEFAULT_BALLUNG, C.BALLUNG_MIN, C.BALLUNG_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "capacity_slider": SettingSpec("capacity", int, C.DEFAULT_CAPACITY, C.CAPACITY_MIN, C.CAPACITY_MAX),
    "moves_select": SettingSpec("moves", _moves_from_str, list(IR.MOVE_TYPES)),
    "budget_select": SettingSpec("budget", _int_choice(C.BUDGETS), C.DEFAULT_BUDGET),
}
PRESET_KEYS = {"n": "n_slider", "ballung": "ballung_slider", "seed": "seed_input", "capacity": "capacity_slider",
               "active_moves": "moves_select", "budget": "budget_select"}
STEPS = {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = int(lo + round((st.session_state[key] - lo) / step) * step)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            if state_key == "moves_select":
                value = _moves_to_str(value)
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
