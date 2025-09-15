# app_cascade.py — NetSafe Cascade Viewer (strict graph types + robust filename parsing)
import os, time, glob, re
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# ========================== Page & Theme ==========================
st.set_page_config(page_title="Cascade Viewer", layout="wide")
st.markdown("""
<style>
:root { --bg: #0e1117; --fg: #e6e6e6; }
.stApp { background: var(--bg); color: var(--fg); }
.block-container { padding-top: .8rem; }
.sidebar .sidebar-content { background: #0e1117; }
</style>
""", unsafe_allow_html=True)

# ========================== Helpers / IO ==========================
# IMPORTANT: use re.search (not match) so uploaded tmp names like "_uploaded_infect_prob_...csv" still parse
CSV_PATTERN = re.compile(
    r"infect_prob_(?P<graph>[a-zA-Z0-9_]+)_(?P<n>\d+)a_(?P<scenario>.+?)\.csv$"
)
# Recognize attackers encoded in different ways
ATTACKER_TOKEN1 = re.compile(r"(?:idx|att|atk|attackers?)[-_]([0-9][0-9\-_,]*)", re.I)
ATTACKER_TOKEN2 = re.compile(r"(?:leaf|leaves)([0-9][0-9\-_,]*)", re.I)  # e.g., leaves2-4

VALID_GRAPHS = {"complete", "pure_star", "star"}

def parse_from_filename(name_or_path: str):
    """
    Parse metadata from a filename like:
      infect_prob_<graph>_<Na>_<scenario>.csv
    Returns: (graph_type:str, n_agents:int|None, attackers:list[int], scenario_str)
    """
    name = os.path.basename(name_or_path)
    m = CSV_PATTERN.search(name)  # search instead of match
    if not m:
        return ("unknown", None, [], name.rsplit(".", 1)[0])

    graph = m.group("graph").strip().lower()
    n = int(m.group("n"))
    scenario = m.group("scenario")

    # Try multiple attacker encodings
    attackers = []
    for token in (ATTACKER_TOKEN1, ATTACKER_TOKEN2):
        mm = token.search(scenario)
        if mm:
            raw = mm.group(1)
            parts = re.split(r"[-_,]", raw)
            attackers = [int(p) for p in parts if p.isdigit()]
            break

    return (graph, n, attackers, scenario)

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("infected_prob", "agent", "round"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["infected_prob"] = df["infected_prob"].fillna(0.0).clip(0, 1)
    df["agent"] = df["agent"].fillna(0).astype(int)
    df["round"] = df["round"].fillna(0).astype(int)
    return df.sort_values(["round", "agent"]).reset_index(drop=True)

def discover_csvs(aggregates_dir: str):
    """Return sorted list of matching CSV paths in a directory."""
    return sorted(glob.glob(os.path.join(aggregates_dir, "infect_prob_*.csv")))

def group_scenarios(paths):
    """Return {graph: {n_agents: {scenario: path}}} built from filename parsing."""
    results = {}
    for f in paths:
        graph, n, _attackers, scenario = parse_from_filename(f)
        if graph not in VALID_GRAPHS or n is None:
            continue
        results.setdefault(graph, {}).setdefault(n, {})[scenario] = f
    return results

# ========================== Graph Builders ==========================
def build_graph(graph_type: str, n: int) -> nx.Graph:
    """Build exactly the three supported graphs by name."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    gt = (graph_type or "").strip().lower()

    if gt == "pure_star":
        # hub 0, leaves 1..n-1
        for i in range(1, n):
            G.add_edge(0, i)

    elif gt == "star":
        # hub 0, leaves connected to hub + ring among leaves
        for i in range(1, n):
            G.add_edge(0, i)
        if n > 2:
            # cycle on leaves: 1-2-3-...-(n-1)-1
            for i in range(1, n - 1):
                G.add_edge(i, i + 1)
            G.add_edge(1, n - 1)

    elif gt == "complete":
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j)

    else:
        # Strict mode: unknown graph -> no edges (UI will warn)
        pass

    return G

def radial_star_layout(n: int):
    """Place node 0 at center, others on a circle."""
    pos = {0: (0.0, 0.0)}
    if n <= 1: return pos
    r = 1.0
    for i in range(1, n):
        th = 2*np.pi*(i-1)/max(1, n-1)
        pos[i] = (r*np.cos(th), r*np.sin(th))
    return pos

def layout_for(G: nx.Graph, graph_type: str):
    gt = (graph_type or "").lower().strip()
    if gt in {"pure_star", "star"}:
        return radial_star_layout(G.number_of_nodes())
    # Deterministic layout for 'complete'
    return nx.spring_layout(G, seed=42, dim=2)

# ========================== Color / Scale ==========================
COLORSCALE = [
    [0.00, "#2b6cb0"],  # low  (blue)
    [0.50, "#f6e05e"],  # mid  (amber)
    [1.00, "#e53e3e"],  # high (red)
]

def compute_global_vmin_vmax(df):
    vals = df["infected_prob"].astype(float).values
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if np.isclose(lo, hi):
        pad = max(1e-3, 0.05 * (hi if hi else 1.0))
        lo, hi = lo - pad, hi + pad
    return lo, hi

# ========================== Plotting ==========================
def plot_network(G, pos, raw_probs, round_idx, vmin, vmax, scenario, attackers):
    nodes = list(G.nodes())
    x = [pos[i][0] for i in nodes]
    y = [pos[i][1] for i in nodes]
    raw_vals = [raw_probs.get(i, 0.0) for i in nodes]

    # Edge coordinates (always from G)
    ex, ey = [], []
    for u, v in G.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color="#6b7280", width=2),
        hoverinfo="none", showlegend=False
    ))

    labels = []
    for i, n in enumerate(nodes):
        tag = " 🛡️" if n in attackers else ""
        labels.append(f"{n}{tag}<br>P={raw_vals[i]:.3f}")

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text",
        text=labels, textposition="top center",
        textfont=dict(color="#e5e7eb"),
        marker=dict(
            size=[48 if n in attackers else 40 for n in nodes],
            line=dict(color="#ffffff", width=[4 if n in attackers else 2 for n in nodes]),
            color=raw_vals,
            colorscale=COLORSCALE,
            cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(
                title=dict(text=f"P(infected) [{vmin:.3f}, {vmax:.3f}]", font=dict(color="#e5e7eb")),
                tickfont=dict(color="#e5e7eb"),
                tickcolor="#e5e7eb",
                outlinecolor="#e5e7eb"
            )
        ),
        hoverinfo="text", showlegend=False
    ))

    fig.update_layout(
        title=f"Round {round_idx} • scenario: {scenario}",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=560, margin=dict(l=8, r=8, t=42, b=8),
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e6e6e6")
    )
    return fig

def plot_lines(df, current_round):
    pivot = df.pivot_table(index="round", columns="agent",
                           values="infected_prob", aggfunc="mean").sort_index()
    pivot["system_avg"] = pivot.mean(axis=1)
    fig = px.line(
        pivot.reset_index(), x="round", y=list(pivot.columns),
        markers=True, template="plotly_dark",
        title="Infection probability per node + system average",
        range_y=[0, 1]  # ✅ Fix y-axis to 0–1
    )
    fig.add_vline(x=current_round, line_width=2, line_dash="dash", line_color="#f6e05e")
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e6e6e6"), height=560,
        legend_title_text=""
    )
    return fig


# ========================== UI / App ==========================
def main():
    st.title("Cascade Viewer")

    # -------- Sidebar: Data source selection --------
    st.sidebar.header("Data Source")
    aggregates_dir = st.sidebar.text_input("Aggregates folder", value="./aggregates")
    found_paths = discover_csvs(aggregates_dir)
    if not found_paths:
        st.sidebar.info("No files like `infect_prob_*.csv` found in this folder.")

    scenarios_map = group_scenarios(found_paths)

    chosen_path = None
    if scenarios_map:
        graphs = sorted(scenarios_map.keys())

        # ✅ Default to "pure_star" if present
        default_graph = "pure_star" if "pure_star" in graphs else graphs[0]
        graph_type = st.sidebar.selectbox(
            "Graph type",
            graphs,
            index=graphs.index(default_graph)
        )

        agents_list = sorted(scenarios_map[graph_type].keys())

        # ✅ Default to 6 agents if available
        default_agents = 6 if 6 in agents_list else agents_list[0]
        n_agents = st.sidebar.selectbox(
            "Number of agents",
            agents_list,
            index=agents_list.index(default_agents)
        )

        available_scenarios = sorted(scenarios_map[graph_type][n_agents].keys())

        # ✅ Prefer scenarios containing "hub" or "leaf"
        default_scenario = next(
            (s for s in available_scenarios if "hub" in s.lower() or "leaf" in s.lower()),
            available_scenarios[0]
        )
        scenario_key = st.sidebar.selectbox(
            "Scenario (from filename)",
            available_scenarios,
            index=available_scenarios.index(default_scenario)
        )

        chosen_path = scenarios_map[graph_type][n_agents][scenario_key]
    else:
        chosen_path = st.sidebar.selectbox("Pick a CSV", found_paths, index=0) if found_paths else None

    if not chosen_path:
        st.error("No CSV selected or found in the aggregates folder.")
        st.stop()

    # -------- Load & parse scenario metadata --------
    df = load_csv(chosen_path)
    parsed_graph, parsed_n, attackers, scenario = parse_from_filename(chosen_path)

    # Validate graph type strictly
    if parsed_graph not in VALID_GRAPHS:
        st.warning(f"Graph type `{parsed_graph}` not one of {sorted(VALID_GRAPHS)} — showing nodes only.")

    # Reconcile agent count with data
    n_in_df = int(df["agent"].max()) + 1 if not df.empty else (parsed_n or 0)
    if parsed_n is None or parsed_n != n_in_df:
        parsed_n = n_in_df

    # Build graph & layout
    G = build_graph(parsed_graph, parsed_n)
    pos = layout_for(G, parsed_graph)

    # Rounds and color scaling
    rounds = sorted(df["round"].unique().tolist())
    if not rounds:
        st.error("CSV has no rounds.")
        st.stop()
    min_r, max_r = int(min(rounds)), int(max(rounds))
    # Always use 0–1 range for infection probability to keep plots comparable
    vmin, vmax = 0.0, 1.0


    # -------- Header summary --------
    display_name = os.path.basename(chosen_path)

    colA, colB, colC, colD, colE, colF = st.columns([2,1,1,2,1,1])
    with colA:
        st.markdown(f"**Data:** `{display_name}`")
    with colB:
        st.markdown(f"**Graph:** `{parsed_graph}`")
    with colC:
        st.markdown(f"**Agents:** `{parsed_n}`")
    with colD:
        att_txt = ", ".join(map(str, attackers)) if attackers else "—"
        st.markdown(f"**Attackers:** `{att_txt}`  •  **Scenario:** `{scenario}`")
    with colE:
        st.markdown(f"**Edges:** `{G.number_of_edges()}`")
    with colF:
        deg0 = G.degree(0) if 0 in G else 0
        st.markdown(f"**deg(0):** `{deg0}`")

    # -------- Layout: two synced plots --------
    c1, c2 = st.columns(2)
    net_ph = c1.empty()
    line_ph = c2.empty()

    # -------- Sidebar: Animation controls --------
    st.sidebar.header("Animation")
    fps = st.sidebar.slider("Speed (frames/sec)", 1, 12, 4, 1)
    play_col, stop_col, reset_col = st.sidebar.columns([1,1,1])
    start = play_col.button("▶ Start")
    stop = stop_col.button("⏸ Stop")
    reset = reset_col.button("⏮ Reset")

    if "playing" not in st.session_state: st.session_state.playing = False
    if "current_round" not in st.session_state: st.session_state.current_round = min_r

    if reset:
        st.session_state.current_round = min_r
        st.session_state.playing = False
    if start:
        st.session_state.playing = True
    if stop:
        st.session_state.playing = False

    r_manual = st.slider(
        "Round", min_value=min_r, max_value=max_r,
        value=int(st.session_state.current_round), step=1, key="round_slider"
    )
    if not st.session_state.playing:
        st.session_state.current_round = r_manual

    # -------- Render function --------
    def render_round(r):
        dfr = df[df["round"] == r]
        raw_probs = {int(a): float(p) for a, p in zip(dfr["agent"], dfr["infected_prob"])}
        for a in range(parsed_n):
            raw_probs.setdefault(a, 0.0)
        net_fig = plot_network(G, pos, raw_probs, r, vmin, vmax, scenario, attackers)
        line_fig = plot_lines(df, r)
        net_ph.plotly_chart(net_fig, use_container_width=True)
        line_ph.plotly_chart(line_fig, use_container_width=True)

    # -------- Animation loop --------
    if st.session_state.playing:
        render_round(st.session_state.current_round)
        nr = st.session_state.current_round + 1
        if nr > max_r: nr = min_r
        st.session_state.current_round = nr
        time.sleep(1.0 / max(1, fps))
        st.rerun()
    else:
        render_round(st.session_state.current_round)

if __name__ == "__main__":
    main()
