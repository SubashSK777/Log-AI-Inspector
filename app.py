"""
app.py  —  Log AI Inspector  |  Premium Streamlit UI
"""

import streamlit as st
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time

from model import LogModel
from utils import preprocess_logs, VOCAB_SIZE, WINDOW_SIZE

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Log AI Inspector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS  ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080c14; color: #e2e8f0; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1221 0%, #0a0f1e 100%);
    border-right: 1px solid rgba(99,179,237,0.12);
}

/* ── metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #0f1729 0%, #141d35 100%);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,179,237,0.15);
}
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: .78rem; color: #94a3b8; letter-spacing: .06em; text-transform: uppercase; margin-top: 4px; }

/* ── status badges ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .05em;
}
.badge-critical { background: rgba(239,68,68,.15); color: #f87171; border: 1px solid rgba(239,68,68,.35); }
.badge-warn     { background: rgba(245,158,11,.15); color: #fbbf24; border: 1px solid rgba(245,158,11,.35); }
.badge-healthy  { background: rgba(34,197,94,.15);  color: #4ade80; border: 1px solid rgba(34,197,94,.35); }

/* ── log viewer ── */
.log-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 14px;
    border-radius: 8px;
    margin: 3px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem;
    transition: background .15s;
}
.log-row:hover { background: rgba(255,255,255,.04); }
.log-critical { background: rgba(239,68,68,.12) !important; border-left: 3px solid #ef4444; }
.log-warn     { background: rgba(245,158,11,.10) !important; border-left: 3px solid #f59e0b; }
.log-normal   { border-left: 3px solid transparent; }
.attn-pill {
    min-width: 56px;
    text-align: center;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: .7rem;
    font-weight: 600;
    background: rgba(255,255,255,.06);
    color: #94a3b8;
    flex-shrink: 0;
}
.attn-pill.hot { background: rgba(239,68,68,.25); color: #f87171; }

/* ── section header ── */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #93c5fd;
    letter-spacing: .04em;
    text-transform: uppercase;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(99,179,237,0.12);
}

/* ── progress / slider tweaks ── */
div[data-baseweb="slider"] { padding-top: 6px; }

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #0f1729 0%, #141d35 100%);
    border: 1px dashed rgba(99,179,237,0.3);
    border-radius: 12px;
    padding: 10px;
}

/* ── scrollable log container ── */
.log-container {
    max-height: 480px;
    overflow-y: auto;
    background: #0a0f1e;
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 12px;
    padding: 12px;
}
.log-container::-webkit-scrollbar { width: 6px; }
.log-container::-webkit-scrollbar-track { background: transparent; }
.log-container::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Model (cached so it isn't rebuilt on every rerun) ─────────────────────────
@st.cache_resource
def load_model():
    m = LogModel(vocab_size=VOCAB_SIZE)
    m.eval()
    return m


# ── Inference helper ──────────────────────────────────────────────────────────
def run_inference(sequences):
    model = load_model()
    scores, attentions = [], []
    with torch.no_grad():
        for seq in sequences:
            x = torch.tensor(seq).unsqueeze(0)          # (1, L)
            score, attn = model(x)
            scores.append(float(score.item()))
            attentions.append(attn.squeeze(-1).squeeze(0).numpy())  # (L,)
    return scores, attentions


# ── Severity helper ───────────────────────────────────────────────────────────
def severity(score: float):
    if score >= 0.70:
        return "CRITICAL", "badge-critical", "#ef4444"
    if score >= 0.40:
        return "WARNING",  "badge-warn",     "#f59e0b"
    return "HEALTHY", "badge-healthy", "#22c55e"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Log AI Inspector")
    st.markdown("<span style='color:#64748b;font-size:.8rem'>GRU · Soft Attention · Plotly</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Settings")
    anomaly_threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.70, 0.05,
                                  help="Sequences above this score are flagged CRITICAL")
    warn_threshold    = st.slider("Warning Threshold", 0.0, 1.0, 0.40, 0.05,
                                  help="Sequences above this score are flagged WARNING")
    attn_threshold    = st.slider("Attention Highlight", 0.05, 0.50, 0.15, 0.01,
                                  help="Log lines above this attention weight are highlighted")
    show_ma           = st.checkbox("Show Moving Average", value=True)
    ma_window         = st.slider("MA Window", 2, 10, 3, disabled=not show_ma)

    st.divider()
    st.markdown("<span style='color:#475569;font-size:.75rem'>Upload a .txt log file to begin analysis</span>",
                unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:24px 0 8px 0;'>
  <h1 style='font-size:2rem;font-weight:700;margin:0;
             background:linear-gradient(90deg,#60a5fa,#a78bfa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    ⚡ Log AI Inspector
  </h1>
  <p style='color:#64748b;margin:6px 0 0 0;font-size:.9rem;'>
    Real-time anomaly detection · GRU encoder · Explainable attention root cause
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Log File (.txt)", type=["txt", "log"],
                             label_visibility="collapsed")

# Demo mode
if not uploaded:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.info("📂 Upload a log file above, or click **Load Demo** to inspect the bundled sample.")
    with col_r:
        demo_btn = st.button("🚀 Load Demo Logs", use_container_width=True)
    if demo_btn:
        try:
            with open("sample_logs.txt", "r") as f:
                raw_lines = f.read().split("\n")
            st.session_state["demo_lines"] = raw_lines
        except FileNotFoundError:
            st.error("`sample_logs.txt` not found — please upload a file manually.")
    if "demo_lines" not in st.session_state:
        st.stop()
    raw_lines = st.session_state["demo_lines"]
else:
    raw_lines = uploaded.read().decode(errors="replace").split("\n")
    st.session_state.pop("demo_lines", None)

# ── Preprocessing + Inference ─────────────────────────────────────────────────
with st.spinner("🔍 Analysing logs…"):
    sequences, raw_windows = preprocess_logs(raw_lines, window=WINDOW_SIZE)
    if not sequences:
        st.error("No usable log lines found. Please upload a non-empty log file.")
        st.stop()
    time.sleep(0.3)   # slight pause for dramatic effect
    scores, attentions = run_inference(sequences)

n_seq      = len(scores)
n_critical = sum(1 for s in scores if s >= anomaly_threshold)
n_warn     = sum(1 for s in scores if warn_threshold <= s < anomaly_threshold)
n_healthy  = n_seq - n_critical - n_warn
max_score  = max(scores)
overall_lbl, overall_cls, _ = severity(max_score)

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

kpi_data = [
    (k1, str(n_seq),                "Sequences",          "#60a5fa"),
    (k2, str(n_critical),           "Critical",           "#f87171"),
    (k3, str(n_warn),               "Warnings",           "#fbbf24"),
    (k4, str(n_healthy),            "Healthy",            "#4ade80"),
    (k5, f"{max_score:.2f}",        "Peak Score",         "#a78bfa"),
]
for col, val, lbl, color in kpi_data:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color}">{val}</div>
          <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Anomaly Timeline ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Anomaly Timeline</div>', unsafe_allow_html=True)

xs = list(range(n_seq))
colors_pts = []
for s in scores:
    lbl2, _, col2 = severity(s)
    colors_pts.append(col2)

fig = go.Figure()

# Threshold bands
fig.add_hrect(y0=anomaly_threshold, y1=1.05, fillcolor="rgba(239,68,68,0.06)",
              line_width=0, annotation_text="Critical Zone",
              annotation_font_color="#f87171", annotation_position="top left")
fig.add_hrect(y0=warn_threshold, y1=anomaly_threshold, fillcolor="rgba(245,158,11,0.05)",
              line_width=0)

# Main score line
fig.add_trace(go.Scatter(
    x=xs, y=scores,
    mode="lines+markers",
    name="Anomaly Score",
    line=dict(color="#60a5fa", width=2),
    marker=dict(color=colors_pts, size=8, line=dict(width=1.5, color="#1e293b")),
    hovertemplate="<b>Seq %{x}</b><br>Score: %{y:.3f}<extra></extra>",
))

# Moving average
if show_ma and n_seq >= ma_window:
    ma = pd.Series(scores).rolling(ma_window, min_periods=1).mean().tolist()
    fig.add_trace(go.Scatter(
        x=xs, y=ma,
        mode="lines",
        name=f"MA({ma_window})",
        line=dict(color="#a78bfa", width=1.5, dash="dot"),
        hoverinfo="skip",
    ))

# Threshold lines
fig.add_hline(y=anomaly_threshold, line_color="rgba(239,68,68,0.6)", line_dash="dash", line_width=1)
fig.add_hline(y=warn_threshold,    line_color="rgba(245,158,11,0.5)", line_dash="dash", line_width=1)

fig.update_layout(
    height=320,
    margin=dict(l=0, r=0, t=10, b=0),
    plot_bgcolor="#0a0f1e",
    paper_bgcolor="#0a0f1e",
    font=dict(family="Inter", color="#94a3b8"),
    xaxis=dict(title="Sequence #", gridcolor="#1e293b", zerolinecolor="#1e293b"),
    yaxis=dict(title="Anomaly Score", range=[0, 1.05], gridcolor="#1e293b", zerolinecolor="#1e293b"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(99,179,237,0.1)", borderwidth=1),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Root-cause explorer ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔥 Root Cause Explorer — Attention View</div>',
            unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2])

with col_left:
    # Sequence selector table
    df_summary = pd.DataFrame({
        "Seq":   xs,
        "Score": [f"{s:.3f}" for s in scores],
        "Status": [severity(s)[0] for s in scores],
    })
    selected_seq = st.slider("Select Sequence", 0, n_seq - 1, 0, key="seq_slider")

    # Score gauge
    sel_score = scores[selected_seq]
    sel_lbl, sel_cls, sel_col = severity(sel_score)

    st.markdown(f"""
    <div class="metric-card" style="margin-top:12px">
      <div style="font-size:.75rem;color:#64748b;letter-spacing:.06em;text-transform:uppercase">
        Sequence {selected_seq}
      </div>
      <div class="metric-value" style="color:{sel_col};font-size:2.4rem;margin:8px 0">{sel_score:.3f}</div>
      <span class="badge {sel_cls}">{sel_lbl}</span>
    </div>""", unsafe_allow_html=True)

    # Attention bar chart
    attn_vals = attentions[selected_seq]
    fig_attn = go.Figure(go.Bar(
        x=[f"L{i}" for i in range(len(attn_vals))],
        y=attn_vals.tolist(),
        marker_color=["#ef4444" if v > attn_threshold else "#3b82f6" for v in attn_vals],
        hovertemplate="Line %{x}<br>Attention: %{y:.3f}<extra></extra>",
    ))
    fig_attn.update_layout(
        height=200, margin=dict(l=0, r=0, t=6, b=0),
        plot_bgcolor="#0a0f1e", paper_bgcolor="#0a0f1e",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(title="Weight", gridcolor="#1e293b"),
        showlegend=False,
    )
    st.markdown("<div style='margin-top:12px;font-size:.8rem;color:#64748b'>Attention Distribution</div>",
                unsafe_allow_html=True)
    st.plotly_chart(fig_attn, use_container_width=True)

with col_right:
    window_lines = raw_windows[selected_seq]
    attn_vals    = attentions[selected_seq]

    html_rows = []
    for i, (line, weight) in enumerate(zip(window_lines, attn_vals)):
        if not line.strip():
            continue
        is_hot  = weight > attn_threshold
        is_warn = "WARN" in line.upper() or "WARNING" in line.upper()
        is_err  = "ERROR" in line.upper() or "CRITICAL" in line.upper() or "FATAL" in line.upper()

        if is_hot and (is_err or weight > attn_threshold * 1.5):
            row_cls  = "log-row log-critical"
            pill_cls = "attn-pill hot"
            icon     = "🔴"
        elif is_warn or (is_hot and weight > attn_threshold):
            row_cls  = "log-row log-warn"
            pill_cls = "attn-pill"
            icon     = "🟡"
        else:
            row_cls  = "log-row log-normal"
            pill_cls = "attn-pill"
            icon     = "⚪"

        safe_line = (line
                     .replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;"))

        html_rows.append(f"""
        <div class="{row_cls}">
          <span style="font-size:.8rem;flex-shrink:0">{icon}</span>
          <span class="{pill_cls}">{weight:.3f}</span>
          <span style="color:#e2e8f0;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                title="{safe_line}">{safe_line}</span>
        </div>""")

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
      <span style="font-size:.85rem;color:#94a3b8">Sequence {selected_seq} — {len(html_rows)} log lines</span>
      <span class="badge {sel_cls}">{sel_lbl}</span>
    </div>
    <div class="log-container">{''.join(html_rows) if html_rows else '<span style="color:#475569">No log lines in this sequence.</span>'}</div>
    """, unsafe_allow_html=True)

# ── Sequence Heatmap overview ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Attention Heatmap — All Sequences</div>',
            unsafe_allow_html=True)

max_len  = max(len(a) for a in attentions)
matrix   = np.zeros((n_seq, max_len))
for i, a in enumerate(attentions):
    matrix[i, :len(a)] = a

fig_heat = go.Figure(go.Heatmap(
    z=matrix,
    colorscale=[[0, "#0a0f1e"], [0.5, "#1e3a5f"], [1, "#ef4444"]],
    hovertemplate="Seq %{y}, Line %{x}<br>Attention: %{z:.3f}<extra></extra>",
    showscale=True,
    colorbar=dict(title="Attention", tickfont=dict(color="#94a3b8"), titlefont=dict(color="#94a3b8")),
))
fig_heat.update_layout(
    height=250,
    margin=dict(l=0, r=0, t=4, b=0),
    plot_bgcolor="#0a0f1e", paper_bgcolor="#0a0f1e",
    font=dict(family="Inter", color="#94a3b8"),
    xaxis=dict(title="Log Line Index", gridcolor="#1e293b"),
    yaxis=dict(title="Sequence #",     gridcolor="#1e293b"),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px 0;color:#334155;font-size:.75rem;">
  Log AI Inspector · GRU + Soft Attention · Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)
