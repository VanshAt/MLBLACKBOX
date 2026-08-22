"""
MLBlackBox Streamlit Dashboard
================================
Real-time visualization of training health, gradient magnitudes,
fault detection, and one-click backtracking.

Run:
    streamlit run dashboard/app.py
    (from the d:/RBU/MLBLACKBOX directory)
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLBlackBox — Neural Network Fault Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background: #0d0f1a; }

    .stApp {
        background: linear-gradient(135deg, #0d0f1a 0%, #111827 100%);
        color: #e2e8f0;
    }

    /* Hero header */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: rgba(99, 102, 241, 0.5); }

    /* Fault card */
    .fault-card-critical {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
    }
    .fault-card-medium {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
    }
    .fault-card-low {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
    }

    /* Code blocks */
    .stCodeBlock { font-family: 'JetBrains Mono', monospace; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.95);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(17, 24, 39, 0.6);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.25) !important;
        color: #a5b4fc !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

AUDIT_LOG = os.path.join("logs", "audit_log.json")
CHECKPOINT_DIR = "checkpoints"


@st.cache_data(ttl=3)
def load_audit_log():
    if not os.path.exists(AUDIT_LOG):
        return []
    try:
        with open(AUDIT_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def list_checkpoints():
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    import glob
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.json"))
    epochs = []
    for f in files:
        try:
            ep = int(os.path.basename(f).replace("checkpoint_epoch_", "").replace(".json", ""))
            epochs.append(ep)
        except ValueError:
            pass
    return sorted(epochs)


def safe(v, fmt=".4f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return format(v, fmt)


def fault_color(severity):
    return {"critical": "#ef4444", "medium": "#f59e0b", "low": "#3b82f6"}.get(
        severity, "#94a3b8"
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 MLBlackBox")
    st.markdown("Neural Network Fault Tracker")
    st.divider()

    st.markdown("**Audit Log**")
    st.markdown(f"`{AUDIT_LOG}`")

    if st.button("🔄 Refresh Data", key="refresh"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("**Backtrack Controls**")
    available_checkpoints = list_checkpoints()
    if available_checkpoints:
        selected_epoch = st.selectbox(
            "Restore to epoch",
            options=available_checkpoints,
            index=len(available_checkpoints) - 1,
        )
        do_backtrack = st.button(f"⏪ Restore to Epoch {selected_epoch}", key="backtrack_btn")
    else:
        st.info("No checkpoints saved yet.")
        do_backtrack = False
        selected_epoch = None

    st.divider()
    st.caption("Built from scratch · Zero ML frameworks")


# ── Main Header ───────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🧠 MLBlackBox</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Real-time neural network fault tracking · '
    'Gradient health monitoring · Root cause analysis</div>',
    unsafe_allow_html=True,
)

history = load_audit_log()

if not history:
    st.info(
        "📂 No audit log found. Start a training run first:\n\n"
        "```python\npython tests/test_iris.py\n```\n\n"
        "Then click **Refresh Data** in the sidebar."
    )
    st.stop()


# ── Key Metrics Row ───────────────────────────────────────────────────────────

latest = history[-1]
n_epochs = len(history)
faults = [r for r in history if r.get("fault_detected")]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Epochs Logged", n_epochs)
with col2:
    st.metric("Latest Loss", safe(latest.get("loss"), ".6f"))
with col3:
    acc = latest.get("accuracy")
    st.metric("Latest Accuracy", f"{acc:.2%}" if acc is not None else "—")
with col4:
    grad = latest.get("gradient", {}).get("global", {}).get("mean")
    st.metric("Gradient Mean", safe(grad, ".4e"))
with col5:
    st.metric("Faults Detected", len(faults), delta=None)


st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Training Curves", "🔍 Gradient Health", "⚠️ Fault Log", "📋 Audit Table"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Training Curves
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    import json as _json

    epochs_list = [r["epoch"] for r in history]
    losses = [r.get("loss", 0) or 0 for r in history]
    accs = [r.get("accuracy") for r in history]
    fault_epochs_set = {r["epoch"] for r in faults}

    # Loss chart
    st.subheader("Loss over Epochs")

    # Build chart data manually (no pandas)
    chart_data_loss = {"Epoch": epochs_list, "Loss": losses}
    # Use st.line_chart with a dict — works without pandas via arrow
    try:
        import pandas as pd
        df_loss = pd.DataFrame({"Loss": losses}, index=epochs_list)
        st.line_chart(df_loss, color="#6366f1")
    except ImportError:
        st.line_chart({"Loss": losses})

    # Mark fault epochs
    if fault_epochs_set:
        fault_ep_str = ", ".join(str(e) for e in sorted(fault_epochs_set))
        st.caption(f"🔴 Faults detected at epochs: **{fault_ep_str}**")

    # Accuracy chart (if available)
    if any(a is not None for a in accs):
        st.subheader("Accuracy over Epochs")
        acc_vals = [a if a is not None else 0.0 for a in accs]
        try:
            df_acc = pd.DataFrame({"Accuracy": acc_vals}, index=epochs_list)
            st.line_chart(df_acc, color="#10b981")
        except NameError:
            st.line_chart({"Accuracy": acc_vals})


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Gradient Health
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Global Gradient Magnitude")

    grad_means = [
        r.get("gradient", {}).get("global", {}).get("mean") or 0.0
        for r in history
    ]
    grad_maxs = [
        r.get("gradient", {}).get("global", {}).get("max") or 0.0
        for r in history
    ]

    try:
        import pandas as pd
        df_grad = pd.DataFrame(
            {"Mean Gradient": grad_means, "Max Gradient": grad_maxs},
            index=epochs_list,
        )
        st.line_chart(df_grad, color=["#8b5cf6", "#ec4899"])
    except (ImportError, NameError):
        st.line_chart({"Mean": grad_means, "Max": grad_maxs})

    # Per-layer gradient table (latest epoch)
    st.subheader("Per-Layer Gradient Stats (Latest Epoch)")
    per_layer = latest.get("gradient", {}).get("per_layer", [])
    if per_layer:
        rows = []
        for ls in per_layer:
            rows.append({
                "Layer": ls.get("layer", "?"),
                "Max": safe(ls.get("max"), ".4e"),
                "Min": safe(ls.get("min"), ".4e"),
                "Mean": safe(ls.get("mean"), ".4e"),
                "Std": safe(ls.get("std"), ".4e"),
                "Has NaN": "⚠️ YES" if ls.get("has_nan") else "✓ No",
            })
        try:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except ImportError:
            st.json(rows)
    else:
        st.info("No per-layer gradient data in audit log.")

    # Weight magnitudes
    st.subheader("Weight Magnitude (Latest Epoch)")
    wm = latest.get("weight_magnitude", {})
    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("Max", safe(wm.get("max"), ".4e"))
    wc2.metric("Min", safe(wm.get("min"), ".4e"))
    wc3.metric("Mean", safe(wm.get("mean"), ".4e"))


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Fault Log
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    if not faults:
        st.success("✅ No faults detected in this training run.")
    else:
        st.markdown(f"**{len(faults)} fault(s) detected**")
        for fault_rec in faults:
            sev = fault_rec.get("fault_detected", "").lower()
            css_class = f"fault-card-{'critical' if 'EXPLOSION' in sev or 'NAN' in sev else 'medium'}"
            st.markdown(
                f'<div class="{css_class}">'
                f'<strong>Epoch {fault_rec["epoch"]}</strong> — '
                f'<span style="color:{fault_color("critical" if "EXPLOSION" in fault_rec.get("fault_detected","") else "medium")}">'
                f'{fault_rec.get("fault_detected", "UNKNOWN")}</span><br>'
                f'Loss: {safe(fault_rec.get("loss"), ".6f")} &nbsp;|&nbsp; '
                f'Gradient mean: {safe(fault_rec.get("gradient", {}).get("global", {}).get("mean"), ".4e")}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Backtrack action
    st.divider()
    st.subheader("⏪ Backtrack")
    if do_backtrack and selected_epoch is not None:
        ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"checkpoint_epoch_{selected_epoch:05d}.json"
        )
        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            st.success(
                f"Checkpoint at epoch **{selected_epoch}** loaded.\n\n"
                f"Loss: `{safe(ckpt.get('loss'), '.6f')}` | "
                f"Timestamp: `{ckpt.get('timestamp', 'N/A')}`\n\n"
                "To apply this checkpoint to a live network, call:\n"
                "```python\nckpt_mgr.load(epoch, network)\n```"
            )
            st.json({"epoch": ckpt["epoch"], "loss": ckpt["loss"], "accuracy": ckpt.get("accuracy")})
        else:
            st.error(f"Checkpoint file not found: {ckpt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 — Audit Table
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Full Audit Log")

    table_rows = []
    for r in history:
        g = r.get("gradient", {}).get("global", {})
        table_rows.append({
            "Epoch": r.get("epoch"),
            "Loss": safe(r.get("loss"), ".6f"),
            "Accuracy": f"{r.get('accuracy'):.2%}" if r.get("accuracy") is not None else "—",
            "Grad Mean": safe(g.get("mean"), ".4e"),
            "Grad Max": safe(g.get("max"), ".4e"),
            "LR": safe(r.get("learning_rate"), ".4f"),
            "Time (ms)": safe(r.get("epoch_time_ms"), ".1f"),
            "Fault": r.get("fault_detected") or "—",
        })

    try:
        import pandas as pd
        df = pd.DataFrame(table_rows)
        # Highlight fault rows
        def highlight_fault(row):
            if row["Fault"] != "—":
                return ["background-color: rgba(239,68,68,0.15)"] * len(row)
            return [""] * len(row)
        st.dataframe(df.style.apply(highlight_fault, axis=1), use_container_width=True, height=500)
    except ImportError:
        st.json(table_rows)

    st.caption(f"Log file: `{AUDIT_LOG}` ({len(history)} records)")
