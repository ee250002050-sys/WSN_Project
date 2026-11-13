"""Interactive Streamlit dashboard for WSN lifetime optimization.

Keep this file beside wsn_lifetime_lp.py.

Run:
    pip install "streamlit>=1.35" scipy numpy pandas plotly streamlit-autorefresh
    streamlit run wsn_dashboard.py

The dashboard is an educational average-flow simulation. It does not model
packet collisions, MAC scheduling, retransmissions, or radio hardware.
"""
from __future__ import annotations

from dataclasses import replace
from collections import deque
import math
import io
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from pathlib import Path

from wsn_lifetime_lp import (
    Network,
    make_network,
    solve_lifetime_lp,
    shortest_path_baseline,
    radio_tx_energy,
)

st.set_page_config(
    page_title="WSN Lifetime Lab",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Drag-and-drop topology component ----------
topology_editor = components.declare_component(
    "wsn_topology_editor",
    path=str(Path(__file__).parent / "topology_editor_component"),
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {background: rgba(120,140,160,.08);
        border: 1px solid rgba(120,140,160,.22); padding: .7rem .9rem;
        border-radius: 12px;}
      section[data-testid="stSidebar"] {border-right: 1px solid rgba(120,140,160,.2);}
      .small-note {font-size: .88rem; opacity: .82;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📡 Wireless Sensor Network Lifetime Lab")
st.caption(
    "Drag-and-drop topology • lifetime optimization • energy simulation • dual sensitivity • reproducible experiments"
)

# ---------- Session state ----------
defaults = {
    "seed": 7,
    "n_sensors": 15,
    "density": 0.28,
    "area_size": 100.0,
    "battery_scale": 1.0,
    "rate_scale": 1.0,
    "network_signature": None,
    "manual_edges": None,
    "manual_positions": None,
    "manual_generation_rates": None,
    "selected_node": 0,
    "sim_time": 0.0,
    "sim_running": False,
    "sim_strategy": "LP optimized",
    "sim_speed": 1.0,
    "last_sim_tick": None,
    "sim_time_slider": 0.0,
    "experiment_results": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------- Helpers ----------
def network_signature(n, seed, density, area):
    return (int(n), int(seed), round(float(density), 4), float(area))


def undirected_edges_from_arcs(arcs, sink):
    edges = set()
    for u, v in arcs:
        if u != sink and v != sink:
            edges.add(tuple(sorted((u, v))))
        elif v == sink:
            edges.add((u, v))
        elif u == sink:
            edges.add((v, u))
    return edges


def make_custom_network(base: Network, positions, edges, battery_scale=1.0, rate_scale=1.0, generation_rates=None):
    """Rebuild directed arcs and distance-dependent costs from editable topology."""
    n_total = len(positions)
    sink = base.sink
    arcs = []
    for a, b in sorted(set(tuple(sorted(e)) if sink not in e else tuple(e) for e in edges)):
        if a == sink:
            # Normalize sink edges to sensor -> sink.
            a, b = b, sink
        if b == sink:
            if a != sink:
                arcs.append((a, sink))
        else:
            if a != sink and b != sink:
                arcs.extend([(a, b), (b, a)])
    arcs = sorted(set(arcs))
    tx_energy = {}
    for i, j in arcs:
        d = float(np.linalg.norm(positions[i] - positions[j]))
        tx_energy[(i, j)] = radio_tx_energy(d)
    return Network(
        positions=np.asarray(positions, dtype=float),
        sink=sink,
        arcs=arcs,
        tx_energy=tx_energy,
        rx_energy=base.rx_energy,
        generation_rate={i: float((generation_rates or base.generation_rate)[i]) * rate_scale
                         for i in base.generation_rate},
        battery={i: float(base.battery[i]) * battery_scale for i in base.battery},
        source=base.source,
    )


def reachable_to_sink(net: Network):
    reverse = {i: [] for i in range(len(net.positions))}
    for u, v in net.arcs:
        reverse.setdefault(v, []).append(u)
    seen = {net.sink}
    queue = deque([net.sink])
    while queue:
        v = queue.popleft()
        for u in reverse.get(v, []):
            if u not in seen:
                seen.add(u)
                queue.append(u)
    return sorted(set(net.battery) - seen)


def solve_both(net: Network):
    disconnected = reachable_to_sink(net)
    if disconnected:
        raise ValueError(
            "These sensors cannot reach the sink: "
            + ", ".join(map(str, disconnected))
            + ". Add links or restore the generated topology."
        )
    lp = solve_lifetime_lp(net)
    baseline = shortest_path_baseline(net)
    return lp, baseline


def shadow_prices(lp, net):
    """SciPy minimizes -T, so negate inequality marginals for max-T sensitivity."""
    marginals = np.asarray(lp["solver_result"].ineqlin.marginals, dtype=float)
    # Rows follow sorted(net.battery), matching solve_lifetime_lp().
    return {i: max(0.0, -float(marginals[k])) for k, i in enumerate(sorted(net.battery))}


def strategy_metrics(net, lp, baseline, strategy):
    if strategy == "LP optimized":
        lifetime = float(lp["lifetime"])
        energy = dict(lp["energy_used"])
        rates = {arc: float(f / lifetime) if lifetime > 0 else 0.0
                 for arc, f in lp["flows"].items()}
    else:
        lifetime = float(baseline["lifetime"])
        rates = dict(baseline["rates"])
        energy = {i: float(baseline["power"][i] * lifetime) for i in net.battery}
    power = {i: (energy[i] / lifetime if lifetime > 0 else 0.0) for i in net.battery}
    return lifetime, energy, rates, power


def build_network_figure(net, rates, energy_fraction, selected_node=None, sim_time=None):
    pos = net.positions
    max_rate = max([v for v in rates.values() if v > 1e-12] or [1.0])
    fig = go.Figure()
    # Links are drawn as separate traces to make direction and intensity clear.
    for (u, v) in net.arcs:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        rate = rates.get((u, v), 0.0)
        active = rate > 1e-10
        fig.add_trace(go.Scatter(
            # The near-transparent midpoint marker makes an edge selectable by click.
            x=[x1, (x1 + x2) / 2, x2], y=[y1, (y1 + y2) / 2, y2],
            mode="lines+markers",
            line=dict(width=1.2 + (4.5 * rate / max_rate if active else 0),
                      color="#0d9488" if active else "rgba(130,140,150,.30)"),
            marker=dict(size=[0, 10, 0], color="rgba(13,148,136,0.015)"),
            customdata=[[u, v], [u, v], [u, v]],
            hovertemplate=f"{u} → {v}<br>Traffic: {rate:.3f} bit/s<extra></extra>",
            showlegend=False,
        ))
        if active:
            # Small midpoint marker makes direction/intensity easier to read.
            fig.add_annotation(
                x=(x1 * .55 + x2 * .45), y=(y1 * .55 + y2 * .45),
                ax=x1, ay=y1, axref="x", ayref="y",
                xref="x", yref="y", showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.4,
                arrowcolor="#0d9488", opacity=.85,
            )
    sensors = [i for i in net.battery if i != net.source]
    fractions = [min(1.0, max(0.0, energy_fraction.get(i, 0.0))) for i in sensors]
    fig.add_trace(go.Scatter(
        x=pos[sensors, 0], y=pos[sensors, 1],
        mode="markers+text", text=[str(i) for i in sensors],
        textposition="middle center",
        marker=dict(
            size=[19 if i == selected_node else 15 for i in sensors],
            color=fractions, colorscale="RdYlGn_r", cmin=0, cmax=1,
            line=dict(color="#263238", width=1.2),
            colorbar=dict(title="Battery<br>used (%)", tickformat=".0%"),
        ),
        customdata=np.array([[i, net.battery[i], net.generation_rate[i], fractions[k]]
                             for k, i in enumerate(sensors)], dtype=float),
        hovertemplate=(
            "Sensor %{customdata[0]:.0f}<br>Battery: %{customdata[1]:.4f} J"
            "<br>Generation: %{customdata[2]:.1f} bit/s"
            "<br>Battery used: %{customdata[3]:.1%}<extra></extra>"
        ),
        name="Sensors",
    ))
    if net.source is not None:
        src = net.source
        fig.add_trace(go.Scatter(
            x=[pos[src, 0]], y=[pos[src, 1]], mode="markers+text",
            text=["SOURCE"], textposition="top center",
            marker=dict(symbol="hexagon", size=27, color="#ea580c",
                        line=dict(color="#7c2d12", width=1.5)),
            customdata=[[src, net.battery[src], net.generation_rate[src]]],
            name="Source", hovertemplate=(
                "SOURCE node %{customdata[0]:.0f}<br>Battery: %{customdata[1]:.4f} J"
                "<br>Generation: %{customdata[2]:.1f} bit/s<extra></extra>"),
        ))
    s = net.sink
    fig.add_trace(go.Scatter(
        x=[pos[s, 0]], y=[pos[s, 1]], mode="markers+text",
        text=["SINK"], textposition="top center",
        marker=dict(symbol="star", size=26, color="#7c3aed",
                    line=dict(color="#35205f", width=1.5)),
        customdata=[[s]], name="Sink", hovertemplate="Sink<extra></extra>",
    ))
    title = "Network topology and active traffic"
    if sim_time is not None:
        title += f" · simulated t = {sim_time:.2f} s"
    fig.update_layout(
        title=title, height=570, margin=dict(l=15, r=15, t=55, b=15),
        xaxis_title="x position (m)", yaxis_title="y position (m)",
        template="plotly_white", clickmode="event+select",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[-5, max(105, float(np.max(pos[:, 0])) + 5)], zeroline=False)
    fig.update_yaxes(range=[-5, max(105, float(np.max(pos[:, 1])) + 5)],
                     scaleanchor="x", scaleratio=1, zeroline=False)
    return fig


def make_energy_figure(net, power, lifetime, sim_time):
    fig = go.Figure()
    horizon = max(lifetime, 1e-9)
    ts = np.linspace(0, horizon, 100)
    for i in sorted(net.battery):
        rem = np.maximum(0, net.battery[i] - power[i] * ts)
        fig.add_trace(go.Scatter(x=ts, y=rem, mode="lines", name=("SOURCE" if i == net.source else f"Sensor {i}"),
                                 line=dict(width=1.8),
                                 hovertemplate="t=%{x:.3f} s<br>Remaining=%{y:.5f} J<extra></extra>"))
    fig.add_vline(x=min(max(sim_time, 0), horizon), line_dash="dash",
                  line_color="#d97706", annotation_text="Current time")
    fig.update_layout(
        title="Battery energy remaining over time (average-power model)",
        xaxis_title="Time (s)", yaxis_title="Remaining battery energy (J)",
        height=430, template="plotly_white", margin=dict(l=10, r=10, t=55, b=10),
        legend_title="Sensor",
    )
    return fig


def rows_for_sensors(net, lp, baseline, strategy):
    lifetime, energy, rates, power = strategy_metrics(net, lp, baseline, strategy)
    prices = shadow_prices(lp, net)
    rows = []
    for i in sorted(net.battery):
        used = energy[i]
        remaining = max(0.0, net.battery[i] - used)
        rows.append({
            "Node": i,
            "Node type": "SOURCE" if i == net.source else "Sensor",
            "Generation (bit/s)": net.generation_rate[i],
            "Battery (J)": net.battery[i],
            "TX energy (J)": sum(net.tx_energy[(u, v)] * rates[(u, v)] * lifetime
                                 for u, v in net.arcs if u == i),
            "RX energy (J)": sum(net.rx_energy * rates[(u, v)] * lifetime
                                 for u, v in net.arcs if v == i),
            "Energy used (J)": used,
            "Remaining (J)": remaining,
            "Battery used (%)": 100.0 * used / net.battery[i] if net.battery[i] else 0.0,
            "LP battery shadow price (s/J)": prices[i],
            "Binding at LP optimum": abs(lp["energy_used"][i] - net.battery[i]) <= max(1e-8, 1e-5 * net.battery[i]),
            "Power (W)": power[i],
        })
    return pd.DataFrame(rows)


# ---------- Sidebar: persistent controls ----------
with st.sidebar:
    st.header("Simulation controls")
    n_sensors = st.slider("Number of sensors", 5, 50, int(st.session_state.n_sensors), 5)
    seed = st.number_input("Random seed", 0, 100000, int(st.session_state.seed), 1)
    density = st.slider("Extra-link probability", 0.0, 0.8, float(st.session_state.density), 0.04)
    area_size = st.slider("Area size (m)", 50, 300, int(st.session_state.area_size), 10)
    battery_scale = st.slider("Battery scale", 0.25, 3.0, float(st.session_state.battery_scale), 0.25)
    rate_scale = st.slider("Traffic-rate scale", 0.25, 3.0, float(st.session_state.rate_scale), 0.25)
    max_link_range = st.slider("Communication range (m; 0 = unlimited)", 0, 300, 0, 5)
    strategy = st.radio("Routing strategy", ["LP optimized", "Shortest-path baseline"])
    if st.button("Generate fresh network", type="primary", use_container_width=True):
        st.session_state.sim_time = 0.0
        st.session_state.sim_running = False
        st.session_state.manual_edges = None
        st.session_state.manual_positions = None
        st.session_state.manual_generation_rates = None
        st.session_state.network_signature = None
        st.rerun()
    st.caption("Controls affect the model. Results are recomputed for the current settings.")

signature = network_signature(n_sensors, seed, density, area_size)
if st.session_state.network_signature != signature or st.session_state.manual_edges is None:
    base = make_network(n_sensors=int(n_sensors), seed=int(seed),
                        area_size_m=float(area_size), extra_link_probability=float(density))
    st.session_state.manual_edges = undirected_edges_from_arcs(base.arcs, base.sink)
    st.session_state.manual_positions = np.array(base.positions, copy=True)
    st.session_state.manual_generation_rates = dict(base.generation_rate)
    st.session_state.network_signature = signature
    st.session_state.sim_time = 0.0
else:
    # Recreate deterministic base parameters; preserve edited positions and topology.
    base = make_network(n_sensors=int(n_sensors), seed=int(seed),
                        area_size_m=float(area_size), extra_link_probability=float(density))

effective_edges = set(st.session_state.manual_edges)
if max_link_range > 0:
    effective_edges = {e for e in effective_edges
                       if float(np.linalg.norm(st.session_state.manual_positions[e[0]] - st.session_state.manual_positions[e[1]])) <= max_link_range}
net = make_custom_network(
    base,
    st.session_state.manual_positions,
    effective_edges,
    battery_scale=float(battery_scale),
    rate_scale=float(rate_scale),
    generation_rates=st.session_state.manual_generation_rates,
)

disconnected = reachable_to_sink(net)
lp = baseline = None
solve_error = None
# Streamlit reruns the script during playback. Cache the optimization result
# in session state and only resolve when the actual network model changes.
model_key = (
    net.positions.tobytes(), tuple(net.arcs),
    tuple(sorted(net.tx_energy.items())),
    tuple(sorted(net.generation_rate.items())),
    tuple(sorted(net.battery.items())), float(net.rx_energy), int(net.sink), int(net.source),
)
if disconnected:
    st.session_state.pop("cached_model_key", None)
    st.session_state.pop("cached_lp", None)
    st.session_state.pop("cached_baseline", None)
elif st.session_state.get("cached_model_key") == model_key:
    lp = st.session_state.get("cached_lp")
    baseline = st.session_state.get("cached_baseline")
else:
    try:
        with st.spinner("Solving LP and shortest-path baseline..."):
            lp, baseline = solve_both(net)
        st.session_state.cached_model_key = model_key
        st.session_state.cached_lp = lp
        st.session_state.cached_baseline = baseline
    except Exception as exc:
        solve_error = str(exc)

# ---------- Five tabs ----------
tabs = st.tabs([
    "01 · Overview",
    "02 · Network Simulator",
    "03 · Energy Simulation",
    "04 · Optimization & Analysis",
    "05 · Experiments & Reports",
])

# ---------- Overview ----------
with tabs[0]:
    st.subheader("Simulation overview")
    if disconnected:
        st.error("Network is disconnected from the sink for sensors: " + ", ".join(map(str, disconnected)))
        st.info("Open Network Simulator and add links to reconnect every sensor to the sink.")
    elif solve_error:
        st.error("Solver failed: " + solve_error)
    else:
        improvement = 100.0 * (lp["lifetime"] - baseline["lifetime"]) / baseline["lifetime"] if baseline["lifetime"] else 0.0
        total_battery = sum(net.battery.values())
        lp_used = sum(lp["energy_used"].values())
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("LP lifetime", f"{lp['lifetime']:.3f} s")
        c2.metric("LP improvement", f"{improvement:+.2f}%")
        c3.metric("Sensors + source / directed links", f"{n_sensors} + 1 / {len(net.arcs)}")
        c4.metric("Total battery energy", f"{total_battery:.4f} J")
        c5.metric("LP energy consumed", f"{lp_used:.4f} J")
        st.write(f"**Solver status:** {lp['solver_result'].message}")
        st.write(f"**Topology:** connected · **SOURCE:** node {net.source} · **SINK:** node {net.sink} · **Radio RX energy:** {net.rx_energy:.3g} J/bit")
        st.caption("LP energy consumed is the sum across sensor batteries over the optimized lifetime.")
        st.subheader("Lifetime comparison")
        fig = go.Figure(data=[go.Bar(
            x=["Shortest-path baseline", "LP optimized"],
            y=[baseline["lifetime"], lp["lifetime"]],
            marker_color=["#64748b", "#0d9488"],
            text=[f"{baseline['lifetime']:.3f}s", f"{lp['lifetime']:.3f}s"],
            textposition="outside",
        )])
        fig.update_layout(height=350, yaxis_title="Lifetime (s)", template="plotly_white",
                          margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Network snapshot")
        _, _, rates, _ = strategy_metrics(net, lp, baseline, strategy)
        _, energy, _, _ = strategy_metrics(net, lp, baseline, strategy)
        fractions = {i: energy[i] / net.battery[i] for i in net.battery}
        st.plotly_chart(build_network_figure(net, rates, fractions), use_container_width=True)

# ---------- Network simulator ----------
with tabs[1]:
    st.subheader("Interactive topology editor")
    st.markdown(
        "Drag nodes to reposition them. Choose **Add link** and click two nodes, "
        "or choose **Remove link** and click an existing connection.\n\n"
        "**Node roles:** orange hexagon = dedicated, battery-powered **SOURCE** "
        "(also generates traffic); blue circles = ordinary sensors (each also "
        "generates traffic); purple diamond = **SINK**."
    )
    if disconnected:
        st.warning("Some sensors cannot currently reach the sink. You can repair the topology directly in the editor.")

    editor_nodes = [
        {"id": int(i), "label": ("SINK" if i == net.sink else "SOURCE" if i == net.source else str(i)),
         "x": float(st.session_state.manual_positions[i, 0]),
         "y": float(st.session_state.manual_positions[i, 1]),
         "is_sink": bool(i == net.sink), "is_source": bool(i == net.source)}
        for i in range(len(net.positions))
    ]
    editor_edges = [list(map(int, e)) for e in sorted(st.session_state.manual_edges)]
    editor_result = topology_editor(
        nodes=editor_nodes, edges=editor_edges, area_size=float(area_size),
        height=590, key="topology_drag_editor", default=None,
    )
    if isinstance(editor_result, dict):
        changed = False
        returned_positions = editor_result.get("positions")
        if isinstance(returned_positions, list) and len(returned_positions) == len(net.positions):
            try:
                updated_positions = np.array([[float(p["x"]), float(p["y"])] for p in returned_positions], dtype=float)
                updated_positions = np.clip(updated_positions, 0.0, float(area_size))
                if not np.allclose(updated_positions, st.session_state.manual_positions):
                    st.session_state.manual_positions = updated_positions
                    changed = True
            except (KeyError, TypeError, ValueError):
                pass
        returned_edges = editor_result.get("edges")
        if isinstance(returned_edges, list):
            normalized_edges = set()
            for edge in returned_edges:
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    continue
                a, b = int(edge[0]), int(edge[1])
                if a == b or not (0 <= a < len(net.positions) and 0 <= b < len(net.positions)):
                    continue
                normalized_edges.add((a, b) if net.sink in (a, b) else tuple(sorted((a, b))))
            if normalized_edges != st.session_state.manual_edges:
                st.session_state.manual_edges = normalized_edges
                changed = True
        selected = editor_result.get("selected_node")
        if selected is not None:
            try:
                st.session_state.selected_node = int(selected)
            except (TypeError, ValueError):
                pass
        if changed:
            st.session_state.sim_time = 0.0
            st.session_state.sim_running = False
            st.rerun()
    st.caption(
        "Editor controls: **Move** = drag nodes · **Add link** = click two nodes · "
        "**Remove link** = click a connection. SOURCE = orange hexagon; SINK = purple diamond."
    )
    sensors_and_sink = list(range(len(net.positions)))
    selected_node = st.selectbox(
        "Select node to move / inspect",
        sensors_and_sink,
        index=min(int(st.session_state.selected_node), len(sensors_and_sink)-1),
        format_func=lambda i: f"Sink ({i})" if i == net.sink else f"SOURCE ({i})" if i == net.source else f"Sensor {i}",
    )
    st.session_state.selected_node = selected_node
    if selected_node != net.sink:
        selected_label = "SOURCE" if selected_node == net.source else f"sensor {selected_node}"
        st.markdown(f"**Selected {selected_label}**")
        if lp is not None:
            sensor_df = rows_for_sensors(net, lp, baseline, strategy)
            detail = sensor_df[sensor_df["Node"] == selected_node].iloc[0]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Battery", f"{detail['Battery (J)']:.4f} J")
            d2.metric("Generation", f"{detail['Generation (bit/s)']:.1f} bit/s")
            d3.metric("TX energy", f"{detail['TX energy (J)']:.5f} J")
            d4.metric("RX energy", f"{detail['RX energy (J)']:.5f} J")
            st.caption(f"Total power: {detail['Power (W)']:.4g} W · LP battery shadow price: {detail['LP battery shadow price (s/J)']:.4g} s/J")
    pos = st.session_state.manual_positions
    colx, coly, colbtn = st.columns([1, 1, 1])
    with colx:
        new_x = st.number_input("X position (m)", 0.0, float(area_size), float(pos[selected_node, 0]), 1.0, key=f"x_{selected_node}")
    with coly:
        new_y = st.number_input("Y position (m)", 0.0, float(area_size), float(pos[selected_node, 1]), 1.0, key=f"y_{selected_node}")
    with colbtn:
        st.write("")
        st.write("")
        if st.button("Apply node position", use_container_width=True):
            pos[selected_node] = [new_x, new_y]
            st.session_state.manual_positions = pos
            st.session_state.sim_time = 0.0
            st.rerun()
    if selected_node != net.sink:
        current_rate = float(st.session_state.manual_generation_rates[selected_node])
        edited_rate = st.number_input("Node data-generation rate (bit/s, before global scale)",
                                      min_value=0.0, max_value=1e7, value=current_rate,
                                      step=50.0, key=f"rate_{selected_node}")
        if st.button("Apply generation-rate change", key=f"apply_rate_{selected_node}"):
            st.session_state.manual_generation_rates[selected_node] = float(edited_rate)
            st.session_state.sim_time = 0.0
            st.rerun()
    st.divider()
    edge_list = sorted(st.session_state.manual_edges)
    with st.expander("Add or remove a connection", expanded=True):
        a, b = st.columns(2)
        with a:
            node_a = st.selectbox("Endpoint A", sensors_and_sink, index=0, key="edge_a",
                                  format_func=lambda i: f"Sink ({i})" if i == net.sink else f"SOURCE ({i})" if i == net.source else f"Sensor {i}")
        with b:
            node_b = st.selectbox("Endpoint B", sensors_and_sink, index=net.sink, key="edge_b",
                                  format_func=lambda i: f"Sink ({i})" if i == net.sink else f"SOURCE ({i})" if i == net.source else f"Sensor {i}")
        if node_a == node_b:
            st.caption("Choose two different nodes.")
        else:
            edge = (node_a, node_b) if net.sink in (node_a, node_b) else tuple(sorted((node_a, node_b)))
            if edge in edge_list:
                if st.button("Remove selected connection", use_container_width=True):
                    st.session_state.manual_edges.remove(edge)
                    st.session_state.sim_time = 0.0
                    st.rerun()
            else:
                if st.button("Add selected connection", use_container_width=True):
                    st.session_state.manual_edges.add(edge)
                    st.session_state.sim_time = 0.0
                    st.rerun()
    st.caption(f"{len(effective_edges)} effective undirected connections · {len(net.arcs)} directed arcs" + (f" · range limit {max_link_range} m" if max_link_range else " · unlimited range"))
    # Link creation/removal is handled inside the custom topology editor.
    # Its updated edge list is returned above and stored in session state.
    if disconnected:
        st.error("Disconnected sensors: " + ", ".join(map(str, disconnected)))
    elif solve_error:
        st.error(solve_error)
    else:
        st.success("Every sensor and the designated SOURCE can reach the sink. LP and baseline results are valid for this topology.")

# ---------- Energy simulation ----------
with tabs[2]:
    st.subheader("Average-flow battery depletion")
    st.warning(
        "This animation converts optimized/shortest-path average traffic rates into average power. "
        "It is not a packet-level MAC simulator and does not model collisions or retransmissions."
    )
    if disconnected or solve_error:
        st.info("Repair the network before running an energy simulation.")
    else:
        lifetime, energy, rates, power = strategy_metrics(net, lp, baseline, strategy)
        horizon = max(float(lifetime), 1e-9)
        st.write(f"**Strategy:** {strategy} · **Model lifetime:** {horizon:.4f} s")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶ Play", use_container_width=True, type="primary"):
                if float(st.session_state.sim_time) >= horizon - 1e-10:
                    st.session_state.sim_time = 0.0
                st.session_state.sim_running = True
                st.session_state.last_sim_tick = time.monotonic()
        with c2:
            if st.button("Ⅱ Pause", use_container_width=True):
                st.session_state.sim_running = False
                st.session_state.last_sim_tick = None
        with c3:
            if st.button("↺ Reset", use_container_width=True):
                st.session_state.sim_time = 0.0
                st.session_state.sim_running = False
                st.session_state.last_sim_tick = None
                st.session_state.sim_time_slider = 0.0
        speed_options = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        current_speed = float(st.session_state.sim_speed)
        if current_speed not in speed_options:
            current_speed = 1.0
        speed = st.select_slider("Playback speed", options=speed_options,
                                 value=current_speed,
                                 format_func=lambda v: f"{v:g}×")
        st.session_state.sim_speed = speed
        # Advance by actual wall-clock time between reruns. This avoids drift and
        # avoids depending on a fixed increment per rerun (which can stall).
        if st.session_state.sim_running:
            st_autorefresh(interval=100, limit=None, key="wsn_energy_autorefresh_v2")
            now = time.monotonic()
            previous = st.session_state.last_sim_tick
            if previous is None:
                previous = now
            elapsed = max(0.0, min(now - previous, 1.0))
            st.session_state.sim_time = min(horizon, float(st.session_state.sim_time) + elapsed * speed)
            st.session_state.last_sim_tick = now
            if st.session_state.sim_time >= horizon - 1e-10:
                st.session_state.sim_running = False
                st.session_state.last_sim_tick = None
        st.session_state.sim_time = min(horizon, max(0.0, float(st.session_state.sim_time)))
        # Show a scrubber only while paused; keeping it out of the active playback
        # rerun loop prevents widget-state synchronization from interrupting Play.
        if not st.session_state.sim_running:
            st.session_state.sim_time_slider = float(st.session_state.sim_time)
            slider_value = st.slider("Simulation time (s)", min_value=0.0, max_value=float(horizon),
                                     value=float(st.session_state.sim_time_slider), key="sim_time_slider_paused")
            st.session_state.sim_time = float(slider_value)
        t = st.session_state.sim_time
        remaining = {i: max(0.0, net.battery[i] - power[i] * t) for i in net.battery}
        fractions_used = {i: 1.0 - remaining[i] / net.battery[i] for i in net.battery}
        exhausted = [i for i in net.battery if remaining[i] <= max(1e-10, net.battery[i] * 1e-8)]
        first_depletion = min((net.battery[i] / power[i] for i in net.battery if power[i] > 0), default=horizon)
        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated time", f"{t:.3f} s")
        m2.metric("Remaining sensor energy", f"{sum(remaining.values()):.5f} J")
        m3.metric("Depleted nodes", str(len(exhausted)))
        st.progress(min(1.0, t / horizon), text=f"{100*t/horizon:.1f}% of modeled lifetime")
        st.plotly_chart(build_network_figure(net, rates, fractions_used, sim_time=t), use_container_width=True)
        st.plotly_chart(make_energy_figure(net, power, horizon, t), use_container_width=True)
        if exhausted:
            st.error("First depleted node(s): " + ", ".join(("SOURCE" if i == net.source else f"Sensor {i}") for i in exhausted))
        else:
            st.info(f"First battery depletion is predicted at t = {first_depletion:.4f} s.")
        st.caption("Battery curves use each node's modeled TX + RX power. Playback advances using measured elapsed time between browser refreshes; 100× is a target speed and may be limited by browser/Streamlit rendering.")

# ---------- Optimization & analysis ----------
with tabs[3]:
    st.subheader("Optimization and sensitivity analysis")
    if disconnected or solve_error:
        st.error("Optimization analysis is unavailable until the topology is connected and solvable.")
    else:
        improvement = 100 * (lp["lifetime"] - baseline["lifetime"]) / baseline["lifetime"] if baseline["lifetime"] else 0
        a, b, c = st.columns(3)
        a.metric("LP lifetime", f"{lp['lifetime']:.5f} s")
        b.metric("Shortest-path lifetime", f"{baseline['lifetime']:.5f} s")
        c.metric("LP lifetime gain", f"{improvement:+.2f}%")
        st.markdown(
            "**Fair comparison:** both methods use the same positions, links, generation rates, battery budgets, "
            "and radio-energy parameters. LP routes can split traffic; the baseline routes each sensor's own traffic "
            "along its minimum-transmission-energy path."
        )
        df = rows_for_sensors(net, lp, baseline, strategy)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.subheader("Battery shadow prices")
        st.markdown(
            "The LP minimizes `-T` internally, so the reported battery shadow price negates the solver's "
            "inequality marginal. Units are seconds per joule and the value is local to the current optimum."
        )
        prices = shadow_prices(lp, net)
        price_fig = go.Figure(go.Bar(
            x=[("SOURCE" if i == net.source else str(i)) for i in prices], y=list(prices.values()),
            marker_color=["#d97706" if lp["energy_used"][i] >= net.battery[i] - max(1e-8, 1e-5*net.battery[i]) else "#64748b"
                          for i in prices],
        ))
        price_fig.update_layout(height=350, xaxis_title="Sensor", yaxis_title="Shadow price (s/J)",
                                template="plotly_white", margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(price_fig, use_container_width=True)
        st.subheader("Complementary-slackness and bottleneck checks")
        slack_rows = []
        for i in sorted(net.battery):
            slack = net.battery[i] - lp["energy_used"][i]
            price = prices[i]
            slack_rows.append({
                "Node": i,
                "Node type": "SOURCE" if i == net.source else "Sensor",
                "Battery slack (J)": slack,
                "Shadow price (s/J)": price,
                "Battery constraint binding": abs(slack) <= max(1e-8, 1e-5 * net.battery[i]),
                "Positive shadow price": price > 1e-8,
                "Interpretation": (
                    "Potential bottleneck: binding and valuable battery"
                    if abs(slack) <= max(1e-8, 1e-5 * net.battery[i]) and price > 1e-8
                    else "Slack battery; zero marginal value"
                    if slack > max(1e-8, 1e-5 * net.battery[i]) and price <= 1e-8
                    else "Degeneracy / alternate optimum / inspect solver tolerances"
                ),
            })
        st.dataframe(pd.DataFrame(slack_rows), use_container_width=True, hide_index=True)
        st.caption("Complementary slackness: a strictly slack battery constraint must have zero shadow price; a binding constraint can still have zero price.")
        st.subheader("Energy burden by sensor")
        burden = df.sort_values("Energy used (J)", ascending=False)
        st.plotly_chart(go.Figure(go.Bar(x=burden["Node"].map(lambda i: "SOURCE" if i == net.source else str(i)), y=burden["Energy used (J)"],
                                         marker_color="#0d9488")).update_layout(
            height=320, xaxis_title="Sensor", yaxis_title="Energy used (J)",
            template="plotly_white", margin=dict(l=10, r=10, t=25, b=10)),
            use_container_width=True)
        st.subheader("LP primal diagnostics")
        eq_resid = np.asarray(lp["solver_result"].eqlin.residual)
        ub_resid = np.asarray(lp["solver_result"].ineqlin.residual)
        st.write(f"Maximum absolute flow-conservation residual: `{np.max(np.abs(eq_resid)):.3e}`")
        st.write(f"Minimum battery slack reported by solver: `{np.min(ub_resid):.3e} J`")
        st.write(f"Solver message: `{lp['solver_result'].message}`")
        result = lp["solver_result"]
        # Equality RHS is zero in this formulation; lower-bound RHS is also zero.
        dual_objective = float(np.dot(np.asarray([net.battery[i] for i in sorted(net.battery)]),
                                      np.asarray(result.ineqlin.marginals)))
        primal_dual_gap = abs(float(result.fun) - dual_objective)
        st.write(f"Primal-dual objective gap: `{primal_dual_gap:.3e}` (solver minimization objective is -T)")
        lower_marginals = np.asarray(result.lower.marginals, dtype=float)[:len(net.arcs)]
        flow_values = np.asarray([lp["flows"][arc] for arc in net.arcs], dtype=float)
        cs_products = np.abs(flow_values * lower_marginals)
        st.write(f"Maximum |flow × lower-bound reduced cost|: `{np.max(cs_products) if len(cs_products) else 0.0:.3e}`")
        reduced_cost_df = pd.DataFrame([
            {"From": arc[0], "To": arc[1], "Flow (total bits)": lp["flows"][arc],
             "Lower-bound reduced cost": float(lower_marginals[k]),
             "Complementarity product": float(cs_products[k])}
            for k, arc in enumerate(net.arcs)
        ])
        with st.expander("Link-level complementary-slackness diagnostics"):
            st.dataframe(reduced_cost_df, use_container_width=True, hide_index=True)
        st.info("Battery shadow prices come from inequality marginals. Flow-conservation dual variables are available in `result.eqlin.marginals`; reduced costs and battery slacks provide complementary-slackness checks.")

# ---------- Experiments and reports ----------
with tabs[4]:
    st.subheader("Repeatable experiments")
    st.markdown("Each experiment uses a reproducible seed. Network parameters and results are included in the downloadable CSV.")
    e1, e2, e3 = st.columns(3)
    with e1:
        seed_start = st.number_input("First seed", min_value=0, max_value=100000, value=int(seed), key="exp_seed")
    with e2:
        seed_count = st.slider("Number of seeds", 1, 30, 5)
    with e3:
        exp_density = st.slider("Experiment extra-link probability", 0.0, 0.8, float(density), 0.04, key="exp_density")
    with st.expander("Sensitivity sweeps (density and battery)"):
        sweep_points = st.slider("Sweep points", 3, 10, 5)
        density_values = np.linspace(0.0, 0.8, int(sweep_points))
        battery_values = np.linspace(0.5, 2.0, int(sweep_points))
        run_density = st.button("Run network-density sweep")
        run_battery = st.button("Run battery-scale sweep")
        if run_density or run_battery:
            sweep_rows = []
            values = density_values if run_density else battery_values
            for k, value in enumerate(values):
                try:
                    dval = float(value) if run_density else float(density)
                    bval = float(battery_scale) if run_density else float(value)
                    sn = make_network(n_sensors=int(n_sensors), seed=int(seed_start) + k,
                                      area_size_m=float(area_size), extra_link_probability=dval)
                    sn = make_custom_network(sn, sn.positions,
                        undirected_edges_from_arcs(sn.arcs, sn.sink), battery_scale=bval,
                        rate_scale=float(rate_scale))
                    slp, sb = solve_both(sn)
                    sweep_rows.append({"Parameter": "Extra-link probability" if run_density else "Battery scale",
                                       "Value": value, "LP lifetime (s)": slp["lifetime"],
                                       "Shortest-path lifetime (s)": sb["lifetime"]})
                except Exception as exc:
                    sweep_rows.append({"Parameter": "Extra-link probability" if run_density else "Battery scale",
                                       "Value": value, "Error": str(exc)})
            sweep_df = pd.DataFrame(sweep_rows)
            st.session_state["sensitivity_results"] = sweep_df
        if st.session_state.get("sensitivity_results") is not None:
            sweep_df = st.session_state["sensitivity_results"]
            st.dataframe(sweep_df, use_container_width=True, hide_index=True)
            if "LP lifetime (s)" in sweep_df:
                fig_sweep = go.Figure()
                fig_sweep.add_trace(go.Scatter(x=sweep_df["Value"], y=sweep_df["LP lifetime (s)"],
                                               mode="lines+markers", name="LP optimized"))
                fig_sweep.add_trace(go.Scatter(x=sweep_df["Value"], y=sweep_df["Shortest-path lifetime (s)"],
                                               mode="lines+markers", name="Shortest path"))
                fig_sweep.update_layout(height=350, xaxis_title="Swept parameter", yaxis_title="Lifetime (s)",
                                        template="plotly_white")
                st.plotly_chart(fig_sweep, use_container_width=True)
                st.download_button("Download sensitivity sweep CSV", sweep_df.to_csv(index=False).encode("utf-8"),
                                    "wsn_sensitivity_sweep.csv", "text/csv")

    if st.button("Run multi-seed experiment", type="primary"):
        result_rows = []
        progress = st.progress(0)
        for idx in range(int(seed_count)):
            this_seed = int(seed_start) + idx
            try:
                experiment_net = make_network(
                    n_sensors=int(n_sensors), seed=this_seed,
                    area_size_m=float(area_size),
                    extra_link_probability=float(exp_density),
                )
                experiment_net = make_custom_network(
                    experiment_net, experiment_net.positions,
                    undirected_edges_from_arcs(experiment_net.arcs, experiment_net.sink),
                    battery_scale=float(battery_scale), rate_scale=float(rate_scale),
                )
                exp_lp, exp_base = solve_both(experiment_net)
                gain = 100 * (exp_lp["lifetime"] - exp_base["lifetime"]) / exp_base["lifetime"] if exp_base["lifetime"] else 0
                result_rows.append({
                    "Seed": this_seed,
                    "Sensors": n_sensors,
                    "Area size (m)": area_size,
                    "Extra-link probability": exp_density,
                    "Battery scale": battery_scale,
                    "Traffic-rate scale": rate_scale,
                    "Directed links": len(experiment_net.arcs),
                    "LP lifetime (s)": exp_lp["lifetime"],
                    "Shortest-path lifetime (s)": exp_base["lifetime"],
                    "Improvement (%)": gain,
                    "Solver status": exp_lp["solver_result"].message,
                })
            except Exception as exc:
                result_rows.append({
                    "Seed": this_seed, "Sensors": n_sensors,
                    "Area size (m)": area_size, "Extra-link probability": exp_density,
                    "Battery scale": battery_scale, "Traffic-rate scale": rate_scale,
                    "Error": str(exc),
                })
            progress.progress((idx + 1) / int(seed_count))
        st.session_state.experiment_results = result_rows
    if st.session_state.experiment_results:
        experiment_df = pd.DataFrame(st.session_state.experiment_results)
        st.dataframe(experiment_df, use_container_width=True, hide_index=True)
        valid = experiment_df[experiment_df.get("LP lifetime (s)", pd.Series(dtype=float)).notna()] if "LP lifetime (s)" in experiment_df else pd.DataFrame()
        if not valid.empty:
            st.subheader("Lifetime across seeds")
            fig = go.Figure()
            fig.add_trace(go.Box(y=valid["LP lifetime (s)"], name="LP optimized", boxpoints="all"))
            fig.add_trace(go.Box(y=valid["Shortest-path lifetime (s)"], name="Shortest path", boxpoints="all"))
            fig.update_layout(height=380, yaxis_title="Lifetime (s)", template="plotly_white",
                              margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.write("**LP lifetime summary**")
            stats = valid["LP lifetime (s)"].agg(["mean", "median", "std", "min", "max"]).to_frame().T
            st.dataframe(stats, use_container_width=True)
            st.caption("Across-seed statistics reduce dependence on one randomly generated topology.")
        csv = experiment_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download experiment CSV", csv, "wsn_experiments.csv", "text/csv")
    else:
        st.info("Choose the number of seeds and run the experiment to populate this section.")

    st.divider()
    st.subheader("Export current run")
    if not disconnected and not solve_error:
        current_df = rows_for_sensors(net, lp, baseline, strategy)
        st.download_button("Download sensor results CSV", current_df.to_csv(index=False).encode("utf-8"),
                           "wsn_sensor_results.csv", "text/csv")
        flows = lp["flows"]
        flow_df = pd.DataFrame([
            {"From": u, "To": v, "LP total bits": f, "LP average rate (bit/s)": f / lp["lifetime"]}
            for (u, v), f in flows.items()
        ])
        st.download_button("Download LP link flows CSV", flow_df.to_csv(index=False).encode("utf-8"),
                           "wsn_link_flows.csv", "text/csv")
        settings = {
            "sensors": n_sensors, "seed": seed, "extra_link_probability": density,
            "area_size_m": area_size, "battery_scale": battery_scale,
            "traffic_rate_scale": rate_scale, "communication_range_m": max_link_range or None, "strategy": strategy,
            "directed_links": len(net.arcs), "source": net.source, "sink": net.sink,
            "source_model": "Dedicated battery-powered SOURCE node plus distributed generation at every sensor",
            "generation_rates_bit_per_s_before_global_scale": st.session_state.manual_generation_rates,
            "lp_lifetime_s": lp["lifetime"], "baseline_lifetime_s": baseline["lifetime"],
            "lp_solver_status": lp["solver_result"].message,
            "model_note": "Average-flow model; not packet-level simulation.",
        }
        st.download_button("Download run settings JSON", json.dumps(settings, indent=2).encode("utf-8"),
                           "wsn_run_settings.json", "application/json")
    else:
        st.info("Reconnect the network to export a valid run.")
