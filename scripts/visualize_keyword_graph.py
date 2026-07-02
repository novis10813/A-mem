"""
visualize_keyword_graph.py
==========================
從 A-MEM cached memory notes 中提取 keywords，
建構 keyword co-occurrence graph 並輸出互動式 HTML 視覺化。

功能：
  1. 單一 snapshot：所有 notes 的完整 keyword graph
  2. 時間演化：按 timestamp 排序，輸出多個時間切片的 graph 比較
  3. Topology 分析：degree centrality、node 數、edge 數隨時間變化

用法：
  uv run --with networkx --with plotly --with pandas \\
      scripts/visualize_keyword_graph.py \\
      --cache-dir artifacts/caches/cached_memories_robust_ollama_llama3.2:latest-v1 \\
      --sample 0 \\
      --output artifacts/output/keyword_graph.html
"""

import argparse
import ast
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Pickle helpers — avoid importing the full A-MEM codebase
# ---------------------------------------------------------------------------

class _Stub:
    """Generic stub that accepts pickle state so attributes are restored."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state: dict):
        self.__dict__.update(state)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


def load_memory_cache(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return SafeUnpickler(f).load()


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def parse_keywords(raw) -> list[str]:
    """Parse keywords from various formats (list, string-of-list, etc.)."""
    if isinstance(raw, list):
        return [str(k).strip().lower() for k in raw if k]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                lst = ast.literal_eval(raw)
                return [str(k).strip().lower() for k in lst if k]
            except Exception:
                pass
        return [k.strip().lower() for k in raw.split(",") if k.strip()]
    return []


def parse_timestamp(ts_str: str) -> datetime | None:
    """Try to parse A-MEM timestamp strings like '1:56 pm on 8 May, 2023'."""
    formats = [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%H:%M on %d %B, %Y",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    return None


def extract_notes(data: dict) -> list[dict]:
    """Convert raw pickle dict to list of clean note dicts."""
    notes = []
    for note_id, note in data.items():
        # Skip notes that have no useful attributes
        if not hasattr(note, "content") and not hasattr(note, "keywords"):
            continue
        kw = parse_keywords(getattr(note, "keywords", []))
        # tags use the same format variety as keywords
        tg = parse_keywords(getattr(note, "tags", []))
        ts = parse_timestamp(str(getattr(note, "timestamp", "")))
        notes.append(
            {
                "id": str(getattr(note, "id", note_id)),
                "content": str(getattr(note, "content", ""))[:200],
                "keywords": kw,
                "tags": tg,
                "timestamp": ts,
                "importance_score": float(getattr(note, "importance_score", 1.0)),
                "retrieval_count": int(getattr(note, "retrieval_count", 0)),
                "category": str(getattr(note, "category", "Uncategorized")),
            }
        )
    # Sort by timestamp if available
    notes.sort(key=lambda n: n["timestamp"] or datetime.min)
    return notes


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "this", "that", "was", "are", "be", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "i", "you", "he", "she", "we", "they",
    "my", "your", "his", "her", "our", "their", "me", "him", "us", "them",
    "carolinesays", "melaniesays", "speaker",  # A-MEM LoCoMo artifacts
}


def build_graph(
    notes: list[dict],
    top_n_keywords: int = 80,
    field: str = "keywords",
) -> nx.Graph:
    """
    Build a term co-occurrence graph from ``field`` ("keywords" or "tags").
    Two terms share an edge if they appear in the same note.
    Edge weight = number of co-occurrences.
    """
    freq: Counter = Counter()
    for n in notes:
        for kw in n.get(field, []):
            if kw not in STOPWORDS and len(kw) > 2:
                freq[kw] += 1

    top_kws = {kw for kw, _ in freq.most_common(top_n_keywords)}

    G = nx.Graph()
    for kw, cnt in freq.items():
        if kw in top_kws:
            G.add_node(kw, freq=cnt)

    for note in notes:
        filtered = [kw for kw in note.get(field, []) if kw in top_kws]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                a, b = filtered[i], filtered[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    return G


# ---------------------------------------------------------------------------
# Plotly graph drawing
# ---------------------------------------------------------------------------

COLOR_EXISTING_EDGE = "#3b4a6b"   # 靛藍偏暗：既有的邊
COLOR_NEW_EDGE      = "#f97316"   # 橙色：新增的邊
COLOR_EXISTING_NODE = "#6366f1"   # 紫色：既有節點
COLOR_NEW_NODE      = "#fb923c"   # 橙色：新增節點


def _layout_positions(G: nx.Graph) -> dict:
    if len(G) == 0:
        return {}
    seed = 42
    try:
        # kamada_kawai gives more evenly spaced nodes than spring_layout
        pos = nx.kamada_kawai_layout(G, weight="weight")
    except Exception:
        try:
            pos = nx.spring_layout(
                G,
                k=2.5 / (len(G) ** 0.4),  # adaptive k: ~2.5 for 80 nodes
                iterations=120,
                seed=seed,
                weight="weight",
            )
        except Exception:
            pos = nx.random_layout(G, seed=seed)
    return pos


def graph_to_plotly(
    G: nx.Graph,
    title: str = "Keyword Co-occurrence Graph",
    highlight_hubs: int = 5,
) -> go.Figure:
    if len(G) == 0:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    pos = _layout_positions(G)
    degree = dict(G.degree())
    centrality = nx.degree_centrality(G)

    # Top hubs
    top_hubs = sorted(centrality, key=centrality.get, reverse=True)[:highlight_hubs]

    # ---- edges ----
    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.7, color="#555"),
        hoverinfo="none",
        showlegend=False,
    )

    # ---- nodes ----
    node_x, node_y, node_text, node_size, node_color, hover_text = (
        [], [], [], [], [], []
    )
    max_deg = max(degree.values()) if degree else 1

    for node in G.nodes():
        x, y = pos[node]
        freq = G.nodes[node].get("freq", 1)
        deg = degree[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node if node in top_hubs else "")
        node_size.append(10 + 30 * (deg / max_deg))
        node_color.append(centrality[node])
        hover_text.append(
            f"<b>{node}</b><br>Frequency: {freq}<br>Degree: {deg}<br>"
            f"Centrality: {centrality[node]:.3f}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color="white"),
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Degree<br>Centrality", thickness=12),
            line=dict(width=1, color="#222"),
        ),
        showlegend=False,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16, color="white")),
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(color="white"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode="closest",
            margin=dict(l=20, r=20, t=60, b=20),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Incremental slider graph  (25 / 50 / 75 / 100 %)
# ---------------------------------------------------------------------------

def build_incremental_slider_figure(
    notes: list[dict],
    top_n_keywords: int = 80,
    quartile_labels: tuple[str, ...] = ("25%", "50%", "75%", "100%"),
    field: str = "keywords",
) -> go.Figure:
    """
    Build a Plotly figure with a slider that shows the co-occurrence graph at
    4 cumulative time slices (25/50/75/100% of notes).
    ``field`` selects which note attribute to use ("keywords" or "tags").
    """
    n = len(notes)
    if n == 0:
        return go.Figure()

    quartile_ends = [max(1, n * q // 100) for q in (25, 50, 75, 100)]

    # --- build the FULL graph first to get a stable layout ---
    G_full = build_graph(notes, top_n_keywords=top_n_keywords, field=field)
    pos = _layout_positions(G_full)

    # For any node not in G_full layout (shouldn't happen), default to 0,0
    def get_pos(node):
        return pos.get(node, (0.0, 0.0))

    # ---- assemble frames ----
    frames = []
    prev_nodes: set[str] = set()
    prev_edges: set[frozenset] = set()

    for label, end in zip(quartile_labels, quartile_ends):
        G = build_graph(notes[:end], top_n_keywords=top_n_keywords, field=field)
        degree = dict(G.degree())
        centrality = nx.degree_centrality(G)
        max_deg = max(degree.values()) if degree else 1
        top_hubs = sorted(centrality, key=centrality.get, reverse=True)[:5]

        cur_nodes = set(G.nodes())
        cur_edges = {frozenset((u, v)) for u, v in G.edges()}
        new_nodes = cur_nodes - prev_nodes
        new_edges = cur_edges - prev_edges

        # ---- edge traces (existing / new) ----
        ex_ex, ey_ex = [], []   # existing edges
        ex_nw, ey_nw = [], []   # new edges
        for u, v in G.edges():
            x0, y0 = get_pos(u)
            x1, y1 = get_pos(v)
            if frozenset((u, v)) in new_edges:
                ex_nw += [x0, x1, None]
                ey_nw += [y0, y1, None]
            else:
                ex_ex += [x0, x1, None]
                ey_ex += [y0, y1, None]

        # ---- node coords (existing / new) ----
        nx_ex, ny_ex, ns_ex, nt_ex, nh_ex = [], [], [], [], []
        nx_nw, ny_nw, ns_nw, nt_nw, nh_nw = [], [], [], [], []
        for node in G.nodes():
            x, y = get_pos(node)
            freq  = G.nodes[node].get("freq", 1)
            deg   = degree[node]
            size  = 10 + 28 * (deg / max_deg)
            label_text = node if node in top_hubs else ""
            hover = (f"<b>{node}</b><br>Freq: {freq}  Degree: {deg}<br>"
                     f"Centrality: {centrality[node]:.3f}<br>"
                     f"<i>{'🆕 New at this slice' if node in new_nodes else '⬛ Existing'}</i>")
            if node in new_nodes:
                nx_nw.append(x); ny_nw.append(y)
                ns_nw.append(size); nt_nw.append(label_text); nh_nw.append(hover)
            else:
                nx_ex.append(x); ny_ex.append(y)
                ns_ex.append(size); nt_ex.append(label_text); nh_ex.append(hover)

        frame_data = [
            # existing edges
            go.Scatter(x=ex_ex, y=ey_ex, mode="lines",
                       line=dict(width=0.6, color=COLOR_EXISTING_EDGE),
                       hoverinfo="none", showlegend=False),
            # new edges
            go.Scatter(x=ex_nw, y=ey_nw, mode="lines",
                       line=dict(width=1.4, color=COLOR_NEW_EDGE),
                       hoverinfo="none", showlegend=False),
            # existing nodes
            go.Scatter(x=nx_ex, y=ny_ex, mode="markers+text",
                       text=nt_ex, textposition="top center",
                       textfont=dict(size=9, color="#a5b4fc"),
                       hovertext=nh_ex, hoverinfo="text",
                       marker=dict(size=ns_ex, color=COLOR_EXISTING_NODE,
                                   opacity=0.75, line=dict(width=1, color="#1e1e3f")),
                       name="Existing", showlegend=True),
            # new nodes
            go.Scatter(x=nx_nw, y=ny_nw, mode="markers+text",
                       text=nt_nw, textposition="top center",
                       textfont=dict(size=9, color="#fed7aa"),
                       hovertext=nh_nw, hoverinfo="text",
                       marker=dict(size=ns_nw, color=COLOR_NEW_NODE,
                                   opacity=0.95, line=dict(width=1.5, color="#7c2d12")),
                       name="New at this slice", showlegend=True),
        ]

        stats = (f"t={label} | Notes: {end}/{n}  "
                 f"Nodes: {len(cur_nodes)} (+{len(new_nodes)})  "
                 f"Edges: {len(cur_edges)} (+{len(new_edges)})")
        frames.append(go.Frame(data=frame_data, name=label,
                               layout=go.Layout(title_text=stats)))

        prev_nodes = cur_nodes
        prev_edges = cur_edges

    # ---- slider steps ----
    steps = [
        dict(
            args=[[f.name], {"frame": {"duration": 0}, "mode": "immediate",
                              "transition": {"duration": 300}}],
            label=f.name,
            method="animate",
        )
        for f in frames
    ]

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="Time slice: ", font=dict(color="white", size=14)),
        pad=dict(t=10, b=10),
        steps=steps,
        bgcolor="#1e1e3f",
        bordercolor="#444",
        font=dict(color="white"),
    )]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=f"Incremental Keyword Graph · t=25% | Notes: {quartile_ends[0]}/{n}",
                font=dict(size=15, color="white"),
            ),
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(color="white"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-1.35, 1.35]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-1.15, 1.15]),
            hovermode="closest",
            legend=dict(
                bgcolor="#1e1e3f", bordercolor="#444", borderwidth=1,
                font=dict(color="white"), x=0.01, y=0.99,
            ),
            sliders=sliders,
            margin=dict(l=20, r=20, t=70, b=80),
            height=700,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.5, xanchor="center",
                    y=-0.08, yanchor="top",
                    buttons=[
                        dict(label="▶ Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 900},
                                         "transition": {"duration": 400},
                                         "fromcurrent": True}]),
                        dict(label="⏸ Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}]),
                    ],
                    bgcolor="#1e1e3f", bordercolor="#555",
                    font=dict(color="white"),
                )
            ],
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Topology evolution chart
# ---------------------------------------------------------------------------

def topology_evolution_chart(
    notes: list[dict],
    slices: int = 8,
    field: str = "keywords",
) -> go.Figure:
    """
    Split notes into `slices` time windows and compute graph stats per window.
    ``field`` selects which note attribute to use ("keywords" or "tags").
    """
    if not notes:
        return go.Figure()

    n = len(notes)
    step = max(1, n // slices)
    checkpoints = list(range(step, n + step, step))

    records = []
    for end in checkpoints:
        subset = notes[:end]

        # True vocabulary: all unique terms seen so far (no cap)
        all_kws: set[str] = set()
        for note in subset:
            for kw in note.get(field, []):
                if kw not in STOPWORDS and len(kw) > 2:
                    all_kws.add(kw)

        # Graph metrics computed on top-60 for consistency
        G = build_graph(subset, top_n_keywords=60, field=field)
        deg = [d for _, d in G.degree()]
        records.append(
            {
                "notes_seen": min(end, n),
                "unique_terms": len(all_kws),
                "edges": G.number_of_edges(),
                "avg_degree": sum(deg) / len(deg) if deg else 0,
                "density": nx.density(G) if G.number_of_nodes() > 1 else 0,
            }
        )

    df = pd.DataFrame(records)

    label_field = "Keyword" if field == "keywords" else "Tag"
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Unique {label_field}s（Vocabulary 成長）", "# Co-occurrence Edges",
            "Average Degree", "Graph Density"
        ],
        vertical_spacing=0.18,
    )

    common = dict(x=df["notes_seen"], mode="lines+markers",
                  marker=dict(size=7))

    fig.add_trace(go.Scatter(y=df["unique_terms"], name=f"Unique {label_field}s",
                             line=dict(color="#7B61FF"), **common), row=1, col=1)
    fig.add_trace(go.Scatter(y=df["edges"], name="Edges",
                             line=dict(color="#00D4AA"), **common), row=1, col=2)
    fig.add_trace(go.Scatter(y=df["avg_degree"], name="Avg Degree",
                             line=dict(color="#FF6B6B"), **common), row=2, col=1)
    fig.add_trace(go.Scatter(y=df["density"], name="Density",
                             line=dict(color="#FFD93D"), **common), row=2, col=2)

    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="Notes Seen", row=row, col=col,
                             color="white", gridcolor="#333")
            fig.update_yaxes(color="white", gridcolor="#333", row=row, col=col)

    fig.update_layout(
        title=dict(
            text=f"{label_field} Graph Topology Evolution",
            font=dict(size=16, color="white"),
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="white"),
        showlegend=False,
        height=550,
        margin=dict(l=40, r=20, t=80, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Top hubs bar chart
# ---------------------------------------------------------------------------

def top_hubs_chart(
    G: nx.Graph,
    top_n: int = 20,
    title: str | None = None,
    colorscale: str = "Plasma",
) -> go.Figure:
    centrality = nx.degree_centrality(G)
    top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, scores = zip(*top) if top else ([], [])
    chart_title = title or f"Top {top_n} Terms by Degree Centrality"

    fig = go.Figure(
        go.Bar(
            x=list(scores),
            y=list(words),
            orientation="h",
            marker=dict(
                color=list(scores),
                colorscale=colorscale,
                showscale=False,
            ),
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=15, color="white")),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="white"),
        yaxis=dict(autorange="reversed", color="white", gridcolor="#333"),
        xaxis=dict(color="white", gridcolor="#333"),
        margin=dict(l=160, r=60, t=60, b=40),
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def build_html(figures: list[tuple[str, go.Figure]], output_path: Path) -> None:
    """Combine multiple plotly figures into a single dark-themed HTML file."""
    html_parts = [
        """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<title>A-MEM Keyword Graph — Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f1117;
    color: #e0e0e0;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    padding: 24px;
  }
  h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 6px;
    background: linear-gradient(90deg, #7B61FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .subtitle {
    color: #888;
    font-size: 0.85rem;
    margin-bottom: 28px;
  }
  .section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #ccc;
    margin: 32px 0 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a2a3a;
  }
  .chart-container {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1e1e2e;
    margin-bottom: 32px;
  }
</style>
</head>
<body>
<h1>A-MEM Keyword Graph</h1>
<p class="subtitle">Visualizing memory note keyword co-occurrence and topology evolution · Built from LoCoMo dataset</p>
"""
    ]

    first = True
    for section_title, fig in figures:
        # Embed full Plotly JS on the first chart; reuse it for subsequent charts
        include_js = True if first else False
        first = False
        div_html = fig.to_html(full_html=False, include_plotlyjs=include_js)
        html_parts.append(f'<p class="section-title">{section_title}</p>')
        html_parts.append(f'<div class="chart-container">{div_html}</div>')

    html_parts.append("</body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"✅ Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize A-MEM keyword graph")
    p.add_argument(
        "--cache-dir",
        default="artifacts/caches/cached_memories_robust_ollama_llama3.2:latest-v1",
        help="Path to cached memory directory, relative to repo root unless absolute",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Which sample index to visualize (0-9), or -1 for all",
    )
    p.add_argument(
        "--top-keywords",
        type=int,
        default=80,
        help="Max number of keywords in the graph",
    )
    p.add_argument(
        "--output",
        default="artifacts/output/keyword_graph.html",
        help="Output HTML path, relative to repo root unless absolute",
    )
    p.add_argument(
        "--slices",
        type=int,
        default=8,
        help="Number of time slices for evolution chart",
    )
    return p.parse_args()


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    args = parse_args()
    cache_dir = repo_path(args.cache_dir)
    output_path = repo_path(args.output)

    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine which pkl files to load
    if args.sample == -1:
        pkl_files = sorted(cache_dir.glob("memory_cache_sample_*.pkl"))
    else:
        pkl_files = [cache_dir / f"memory_cache_sample_{args.sample}.pkl"]

    print(f"📂 Loading {len(pkl_files)} pkl file(s)...")
    all_notes = []
    for pkl in pkl_files:
        print(f"   {pkl.name} ...", end=" ", flush=True)
        raw = load_memory_cache(pkl)
        notes = extract_notes(raw)
        all_notes.extend(notes)
        print(f"{len(notes)} notes")

    print(f"📝 Total notes: {len(all_notes)}")

    # Sort all notes by timestamp
    all_notes.sort(key=lambda n: n["timestamp"] or datetime.min)

    # Build full keyword graph
    print("🔗 Building keyword (purple) graph...")
    G_full = build_graph(all_notes, top_n_keywords=args.top_keywords, field="keywords")
    print(f"   Nodes: {G_full.number_of_nodes()}, Edges: {G_full.number_of_edges()}")

    # Build full tag graph
    print("🔗 Building tag (green) graph...")
    G_tags = build_graph(all_notes, top_n_keywords=args.top_keywords, field="tags")
    print(f"   Nodes: {G_tags.number_of_nodes()}, Edges: {G_tags.number_of_edges()}")

    # Build incremental slider graphs
    print("🎞  Building incremental slider — keywords...")
    fig_slider_kw = build_incremental_slider_figure(
        all_notes, top_n_keywords=args.top_keywords, field="keywords"
    )
    print("🎞  Building incremental slider — tags...")
    fig_slider_tags = build_incremental_slider_figure(
        all_notes, top_n_keywords=args.top_keywords, field="tags"
    )

    # Figures
    figures = [
        # ---- Keywords (purple) ----
        (
            f"全量 Keyword Graph（{G_full.number_of_nodes()} 個關鍵詞，{G_full.number_of_edges()} 條邊）",
            graph_to_plotly(G_full, title="Full Keyword Co-occurrence Graph"),
        ),
        (
            "Incremental Keyword Graph — 拉桿切換 25 / 50 / 75 / 100% 筆記"
            "（<span style='color:#fb923c'>橙色</span>＝本時間點新增，"
            "<span style='color:#818cf8'>靛藍</span>＝已存在）",
            fig_slider_kw,
        ),
        (
            "Top 20 Hub Keywords（Degree Centrality）",
            top_hubs_chart(G_full, top_n=20,
                           title="Top 20 Keywords by Degree Centrality",
                           colorscale="Plasma"),
        ),
        (
            "Keyword Topology 演化：Graph 結構隨對話輪次的變化",
            topology_evolution_chart(all_notes, slices=args.slices, field="keywords"),
        ),
        # ---- Tags (green) ----
        (
            f"全量 Tag Graph（{G_tags.number_of_nodes()} 個標籤，{G_tags.number_of_edges()} 條邊）",
            graph_to_plotly(G_tags, title="Full Tag Co-occurrence Graph"),
        ),
        (
            "Incremental Tag Graph — 拉桿切換 25 / 50 / 75 / 100% 筆記"
            "（<span style='color:#fb923c'>橙色</span>＝本時間點新增，"
            "<span style='color:#818cf8'>靛藍</span>＝已存在）",
            fig_slider_tags,
        ),
        (
            "Top 20 Hub Tags（Degree Centrality）",
            top_hubs_chart(G_tags, top_n=20,
                           title="Top 20 Tags by Degree Centrality",
                           colorscale="Teal"),
        ),
        (
            "Tag Topology 演化：Graph 結構隨對話輪次的變化",
            topology_evolution_chart(all_notes, slices=args.slices, field="tags"),
        ),
    ]

    build_html(figures, output_path)


if __name__ == "__main__":
    main()
