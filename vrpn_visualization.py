"""Plotly-Abbildungen der VRP-Nachbarschaften-Demo: Karten (mehrfarbig je Route), Ablations-Balken, Sweeps,
Skalierung. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import vrpn_constants as C

TOUR_COLOR = "#4c78a8"
IMPROVE_COLOR = "#54a24b"
ROUTE_COLORS = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac"]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def build_instance(xy):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=TOUR_COLOR, line=dict(width=1, color="white")), name="Stopps", hovertemplate="Stopp %{customdata}<extra></extra>", customdata=np.arange(1, len(xy))))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_routes(xy, routes):
    """Jede Route in einer eigenen Farbe (Depot an beiden Enden jeder Route)."""
    fig = go.Figure()
    for i, route in enumerate(routes):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        t = np.concatenate([[0], route, [0]])
        fig.add_trace(go.Scatter(x=xy[t, 0], y=xy[t, 1], mode="lines+markers",
                                  line=dict(color=color, width=2.5), marker=dict(size=6, color=color, line=dict(width=1, color="white")),
                                  name=f"Route {i + 1} ({len(route)} Stopps)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=16, symbol="star", color="#222", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_ablation_bar(labels, improvements):
    fig = go.Figure(go.Bar(x=labels, y=improvements, marker_color=IMPROVE_COLOR))
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Verbesserung ggü. Konstruktion (%)")
    return _base(fig, 340)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    upper = [r["improvement"] + r["improvement_sd"] for r in rows]
    lower = [max(0.0, r["improvement"] - r["improvement_sd"]) for r in rows]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(84,162,75,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["improvement"] for r in rows], mode="lines+markers", line=dict(color=IMPROVE_COLOR, width=2.5), name="Verbesserung ggü. Konstruktion"))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Verbesserung (%)")
    return _base(fig, 380)


def build_scaling(blocks):
    fig = go.Figure()
    colors = (IMPROVE_COLOR, "#f58518")
    for block, color in zip(blocks, colors):
        label, rows = block["label"], block["rows"]
        xs = [r["value"] for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=[r["improvement"] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=label))
    fig.update_xaxes(title_text="Stopps")
    fig.update_yaxes(title_text="Verbesserung (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)
