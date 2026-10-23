"""
MLBlackBox Streamlit Dashboard — Control Room Edition
======================================================
Modern, dark, professional ML observability console for PyTorch neural network training.
"""

import sys
import os
import json
import math
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from tracker.layer_risk import rank_layers_by_risk
from tracker.health_score import calculate_health_score
from runs.session import RunSessionManager
from reports.report_generator import (
    export_markdown_report,
    export_metrics_csv,
    export_fault_log_json,
    export_health_summary_html,
)
from ai.training_doctor import diagnose_run, LocalTrainingDoctor, LLMTrainingDoctor
from ai.failure_forecaster import FailureForecaster, FailureForecast, ForecastRiskLevel
from dashboard.styles import (
    COLORS,
    CONTROL_ROOM_CSS,
    render_stat_box,
    render_status_badge,
    render_alert_box,
    render_health_card,
    render_forecast_banner,
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLBlackBox · AI Control Room",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CONTROL_ROOM_CSS, unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────────
def safe(v, fmt=".4f"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)

session_mgr = RunSessionManager()
DEFAULT_AUDIT_LOG = os.path.join("logs", "audit_log.json")
DEFAULT_CHECKPOINT_DIR = "checkpoints"


# ── Persistent Control Room Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.25rem;">
        <span style="font-size: 1.6rem;">🛡️</span>
        <span style="font-size: 1.35rem; font-weight: 800; color: #22D3EE; letter-spacing: -0.02em;">MLBlackBox</span>
    </div>
    <div style="color: #8B949E; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem;">
        AI Training Flight Control
    </div>
    """, unsafe_allow_html=True)

    # 1. Active Run Selector
    st.markdown('<div class="cr-stat-label">ACTIVE RUN SESSION</div>', unsafe_allow_html=True)
    runs_list = session_mgr.list_runs()
    selected_run = None
    audit_file_to_load = DEFAULT_AUDIT_LOG
    ckpt_dir_to_use = DEFAULT_CHECKPOINT_DIR

    if runs_list:
        run_options = {f"{r['run_id']} ({r.get('status', 'running')})": r for r in runs_list}
        selected_label = st.selectbox("Select Session", options=list(run_options.keys()), label_visibility="collapsed")
        selected_run = run_options[selected_label]
        if selected_run.get("log_path") and os.path.exists(selected_run["log_path"]):
            audit_file_to_load = selected_run["log_path"]
        if selected_run.get("checkpoint_dir") and os.path.exists(selected_run["checkpoint_dir"]):
            ckpt_dir_to_use = selected_run["checkpoint_dir"]
    else:
        st.caption("No registered runs yet. Active telemetry fallback.")

    # 2. External Log Drop
    with st.expander("📁 Upload External Audit Log", expanded=False):
        uploaded_file = st.file_uploader(
            "Drop audit_log.json",
            type=["json"],
            key="side_uploader",
            label_visibility="collapsed"
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # 3. Sidebar Navigation Radio
    st.markdown('<div class="cr-stat-label">CONSOLE NAVIGATION</div>', unsafe_allow_html=True)
    PAGES = [
        "1. Overview",
        "2. Training Metrics",
        "3. Layer Health",
        "4. Fault Timeline",
        "5. Checkpoints",
        "6. Run History",
        "7. AI Training Doctor",
        "8. Failure Forecast",
        "9. Settings",
    ]
    current_page = st.radio("Navigation", options=PAGES, index=0, label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)

    # 4. Quick Actions
    st.markdown('<div class="cr-stat-label">QUICK ACTIONS</div>', unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with c_btn2:
        if st.button("📄 Export", use_container_width=True):
            st.session_state["nav_to_exports"] = True
            st.info("Exports available in tab #9 or Reports page.")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.caption("MLBlackBox v3.2 · Control Room Console")


# ── Load History Telemetry ────────────────────────────────────────────────────
history = []
if "uploaded_file" in locals() and uploaded_file is not None:
    try:
        history = json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded log: {e}")
elif os.path.exists(audit_file_to_load):
    try:
        with open(audit_file_to_load, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception as e:
        st.sidebar.warning(f"Could not load log: {e}")

# Handle Empty State Gracefully
if not history:
    st.markdown("""
    <div class="cr-hero">
        <div>
            <div class="cr-hero-title">🛡️ MLBlackBox Observability Control Room</div>
            <div class="cr-hero-subtitle">Real-time deep learning flight recorder & forensic telemetry</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(render_alert_box(
        "No Active Telemetry Detected",
        "No telemetry logs were found in the selected session or logs/ directory. Run a demo script or training pipeline to populate telemetry.",
        level="info"
    ), unsafe_allow_html=True)

    st.markdown("""
    <div class="cr-card">
        <h4 style="margin-top: 0; color: #22D3EE;">Quick Start Training Verification</h4>
        <p style="color: #8B949E; font-size: 0.9rem;">Launch any of the isolated demo scenarios to observe the control room in action:</p>
        <pre style="background: #161B22; padding: 1rem; border-radius: 8px; border: 1px solid #263041; color: #22D3EE;">
python demo_models/demo_explosion.py --fresh
python demo_models/demo_vanishing.py --fresh
python demo_models/demo_nan.py --fresh
python tests/test_iris.py</pre>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Telemetry Precomputations ─────────────────────────────────────────────────
latest = history[-1]
n_epochs = len(history)
faults = [r for r in history if r.get("fault_detected")]
layer_risks = rank_layers_by_risk(history)

# Health Score
health_score_val = latest.get("health_score")
health_exp_str = latest.get("health_explanation")
if health_score_val is None:
    score, exp, _ = calculate_health_score(latest, history)
    health_score_val = score
    health_exp_str = exp

# Forecast Precursor
latest_fc = latest.get("forecast")
if not latest_fc:
    forecaster = FailureForecaster()
    latest_fc = forecaster.forecast(latest, history, layer_risks).to_dict()

fc_level = latest_fc.get("overall_risk_level", "Low")
fc_score = latest_fc.get("overall_risk_score", 0)
fc_threat = latest_fc.get("likely_failure_type", "None")
fc_epochs = latest_fc.get("estimated_epochs_to_failure")
fc_signals = latest_fc.get("warning_signals", [])
fc_action = latest_fc.get("recommended_action", {})

run_status = selected_run.get("status", "running") if selected_run else ("fault_detected" if faults else "completed")


# ── Page Header / Breadcrumbs ─────────────────────────────────────────────────
page_titles = {
    "1. Overview": "Control Room Overview",
    "2. Training Metrics": "Training Convergence & Dynamics",
    "3. Layer Health": "Per-Layer Risk & Gradient Forensics",
    "4. Fault Timeline": "Fault Detection & Forensic Timeline",
    "5. Checkpoints": "Checkpoint Lifecycle & Comparison",
    "6. Run History": "Historical Training Sessions",
    "7. AI Training Doctor": "AI Diagnostic Assistant",
    "8. Failure Forecast": "Early-Warning Failure Forecaster",
    "9. Settings": "Telemetry & Observability Settings",
}

st.markdown(f"""
<div class="cr-hero">
    <div>
        <div class="cr-hero-title">{page_titles.get(current_page, 'Observability Console')}</div>
        <div class="cr-hero-subtitle">Session: <code style="color: #22D3EE;">{selected_run['run_id'] if selected_run else 'live_session'}</code> &nbsp;·&nbsp; Status: {render_status_badge(run_status, 'critical' if faults else 'success')}</div>
    </div>
    <div style="text-align: right;">
        <span style="font-size: 0.8rem; color: #8B949E;">LATEST LOGGED EPOCH</span><br>
        <span style="font-size: 1.4rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #22D3EE;">{latest.get('epoch', n_epochs)}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if current_page == "1. Overview":
    # 1. Top Banners (Health Card & Forecast Card)
    col_banner1, col_banner2 = st.columns([1, 1])
    with col_banner1:
        st.markdown(render_health_card(health_score_val, health_exp_str), unsafe_allow_html=True)
    with col_banner2:
        st.markdown(render_forecast_banner(
            fc_level, fc_score, fc_threat, fc_epochs, fc_signals[0] if fc_signals else ""
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

    # 2. Key Telemetry Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(render_stat_box("TOTAL EPOCHS", str(n_epochs), "Logged epochs"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_stat_box("LATEST LOSS", safe(latest.get("loss"), ".5f"), "Cross entropy/MSE"), unsafe_allow_html=True)
    with m3:
        acc_s = f"{latest.get('accuracy'):.2%}" if latest.get('accuracy') is not None else "—"
        st.markdown(render_stat_box("ACCURACY", acc_s, "Validation score"), unsafe_allow_html=True)
    with m4:
        grad_mean = latest.get("gradient", {}).get("global", {}).get("mean")
        st.markdown(render_stat_box("GRADIENT MEAN", safe(grad_mean, ".3e"), "Global L2 norm"), unsafe_allow_html=True)
    with m5:
        f_badge_col = COLORS["critical"] if faults else COLORS["success"]
        st.markdown(render_stat_box("FAULTS DETECTED", str(len(faults)), "Verified halts", f_badge_col), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

    # 3. Compact Curves & Risky Layers Split
    c_left, c_right = st.columns([2, 1])

    with c_left:
        st.markdown('<div class="cr-card"><div class="cr-stat-label">LOSS & CONVERGENCE TRAJECTORY</div>', unsafe_allow_html=True)
        epochs = [r.get("epoch", i) for i, r in enumerate(history)]
        losses = [r.get("loss", 0.0) for r in history]
        df_curves = pd.DataFrame({"Epoch": epochs, "Loss": losses}).set_index("Epoch")
        st.line_chart(df_curves, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="cr-card"><div class="cr-stat-label">TOP AT-RISK LAYERS</div>', unsafe_allow_html=True)
        if layer_risks:
            for lr in layer_risks[:3]:
                r_score = lr.get("risk_score", 0)
                badge_lvl = "critical" if r_score >= 70 else ("warning" if r_score >= 40 else "success")
                st.markdown(f"""
                <div style="background: #161B22; border: 1px solid #263041; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div style="font-weight: 700; font-size: 0.9rem; color: #E6EDF3;">{lr.get('layer')}</div>
                        <div style="font-size: 0.75rem; color: #8B949E;">{lr.get('likely_issue')}</div>
                    </div>
                    <div>
                        {render_status_badge(f"{r_score}/100", badge_lvl)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No per-layer gradient risks observed.")
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: TRAINING METRICS
# ==============================================================================
elif current_page == "2. Training Metrics":
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        epoch_range = st.slider("Epoch Range Filter", 0, max(1, n_epochs - 1), (0, max(1, n_epochs - 1)))
    with col_filter2:
        metrics_to_show = st.multiselect("Active Curves", ["Loss", "Accuracy", "Gradient Mean", "Weight Norm"], default=["Loss", "Accuracy"])
    st.markdown('</div>', unsafe_allow_html=True)

    filtered_history = [r for r in history if epoch_range[0] <= r.get("epoch", 0) <= epoch_range[1]]
    f_epochs = [r.get("epoch", i) for i, r in enumerate(filtered_history)]

    # 1. Primary Loss and Accuracy Curves
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown('<div class="cr-card"><div class="cr-stat-label">LOSS DYNAMICS</div>', unsafe_allow_html=True)
        df_loss = pd.DataFrame({"Epoch": f_epochs, "Loss": [r.get("loss") for r in filtered_history]}).set_index("Epoch")
        st.line_chart(df_loss, height=260)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_m2:
        st.markdown('<div class="cr-card"><div class="cr-stat-label">ACCURACY TRAJECTORY</div>', unsafe_allow_html=True)
        acc_data = [r.get("accuracy") for r in filtered_history]
        if any(a is not None for a in acc_data):
            df_acc = pd.DataFrame({"Epoch": f_epochs, "Accuracy": acc_data}).set_index("Epoch")
            st.line_chart(df_acc, height=260)
        else:
            st.info("Accuracy metric not configured for this regression or unsupervised run.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Gradient Norm & Weight Norm Curves
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.markdown('<div class="cr-card"><div class="cr-stat-label">GLOBAL GRADIENT L2 NORM</div>', unsafe_allow_html=True)
        g_means = [r.get("gradient", {}).get("global", {}).get("mean") for r in filtered_history]
        df_grad = pd.DataFrame({"Epoch": f_epochs, "Gradient Norm": g_means}).set_index("Epoch")
        st.line_chart(df_grad, height=240)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_g2:
        st.markdown('<div class="cr-card"><div class="cr-stat-label">WEIGHT MAGNITUDE DRIFT</div>', unsafe_allow_html=True)
        w_means = [r.get("weight_magnitude", {}).get("global", {}).get("mean") for r in filtered_history]
        df_w = pd.DataFrame({"Epoch": f_epochs, "Weight Norm Mean": w_means}).set_index("Epoch")
        st.line_chart(df_w, height=240)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: LAYER HEALTH
# ==============================================================================
elif current_page == "3. Layer Health":
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.markdown('<div class="cr-stat-label">PER-LAYER FAILURE RISK MATRIX</div>', unsafe_allow_html=True)

    if layer_risks:
        df_risk_table = pd.DataFrame(layer_risks)
        st.dataframe(df_risk_table, use_container_width=True)
    else:
        st.info("No per-layer gradient statistics found in telemetry.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Top 3 Alert Cards
    if layer_risks:
        st.markdown("### 🚨 High Risk Layer Diagnostics")
        r_cols = st.columns(min(3, len(layer_risks)))
        for col, lr in zip(r_cols, layer_risks[:3]):
            with col:
                score_num = lr.get("risk_score", 0)
                lvl = "critical" if score_num >= 70 else ("warning" if score_num >= 35 else "info")
                st.markdown(f"""
                <div class="cr-card" style="border-top: 3px solid {COLORS.get(lvl, COLORS['accent_cyan'])};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 800; font-size: 1rem; color: #22D3EE;">{lr.get('layer')}</span>
                        {render_status_badge(lr.get('trend', 'STABLE'), lvl)}
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; font-family: 'JetBrains Mono'; margin-bottom: 0.5rem;">{score_num}<span style="font-size: 0.9rem; color: #8B949E;">/100</span></div>
                    <div style="font-size: 0.85rem; color: #E6EDF3; margin-bottom: 0.35rem;"><strong>Issue:</strong> {lr.get('likely_issue')}</div>
                    <div style="font-size: 0.8rem; color: #8B949E;"><strong>Action:</strong> {lr.get('recommended_action')}</div>
                </div>
                """, unsafe_allow_html=True)

    # Gradient Trends by Layer
    st.markdown('<div class="cr-card"><div class="cr-stat-label">LAYER GRADIENT NORM EVOLUTION</div>', unsafe_allow_html=True)
    per_layer_history = {}
    for r in history:
        ep = r.get("epoch", 0)
        for lstat in r.get("gradient", {}).get("per_layer", []):
            lname = str(lstat.get("layer"))
            if lname not in per_layer_history:
                per_layer_history[lname] = {}
            per_layer_history[lname][ep] = lstat.get("mean", 0.0)

    if per_layer_history:
        df_per_l = pd.DataFrame(per_layer_history)
        df_per_l.index.name = "Epoch"
        st.line_chart(df_per_l, height=280)
    else:
        st.caption("Layer-level gradients are uniform or not recorded.")
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: FAULT TIMELINE
# ==============================================================================
elif current_page == "4. Fault Timeline":
    st.markdown('<div class="cr-stat-label">FAULT DETECTIONS & ROLLBACK EVENTS</div>', unsafe_allow_html=True)

    if faults:
        for idx, f_rec in enumerate(faults, 1):
            f_type = f_rec.get("fault_type", "UNKNOWN_FAULT")
            f_epoch = f_rec.get("epoch", "?")
            f_sev = f_rec.get("severity", "critical")
            badge_lvl = "critical" if f_sev == "critical" else "warning"

            st.markdown(f"""
            <div class="cr-card" style="border-left: 4px solid {COLORS['critical']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.3rem;">⚠️</span>
                        <span style="font-weight: 800; font-size: 1.1rem; color: #EF4444;">{f_type}</span>
                        {render_status_badge(f"Epoch {f_epoch}", "critical")}
                    </div>
                    {render_status_badge(f_sev, badge_lvl)}
                </div>
                <div style="margin-top: 0.75rem; color: #E6EDF3; font-size: 0.9rem;">
                    <strong>Evidence:</strong> <code>{json.dumps(f_rec.get('evidence', {}))}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(render_alert_box(
            "Zero Fault Events Detected",
            f"All {n_epochs} epochs executed cleanly without gradient explosion, vanishing, NaN/Inf, or stagnation triggers.",
            level="success"
        ), unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: CHECKPOINTS
# ==============================================================================
elif current_page == "5. Checkpoints":
    from tracker.checkpoint import CheckpointManager
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir_to_use)
    all_ckpts = ckpt_mgr.list_checkpoints()

    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="cr-stat-label">SNAPSHOT INVENTORY · {ckpt_dir_to_use}</div>', unsafe_allow_html=True)

    if all_ckpts:
        df_ck = pd.DataFrame(all_ckpts)
        st.dataframe(df_ck, use_container_width=True)

        col_ck1, col_ck2 = st.columns(2)
        available_eps = [c["epoch"] for c in all_ckpts]

        with col_ck1:
            st.markdown("**Restore or Delete Checkpoint**")
            target_ep = st.selectbox("Select Target Epoch", options=available_eps, key="ckpt_sel_page")
            b_r1, b_r2 = st.columns(2)
            with b_r1:
                if st.button(f"⏪ Roll Back to Ep {target_ep}", key="btn_r_exec", type="primary"):
                    st.success(f"Restore command ready in training runtime:\n`trainer.resume({target_ep})`")
            with b_r2:
                if st.button(f"🗑️ Delete Ep {target_ep}", key="btn_del_exec"):
                    ckpt_mgr.delete_checkpoint(target_ep)
                    st.warning(f"Epoch {target_ep} checkpoint deleted.")
                    st.rerun()

        with col_ck2:
            st.markdown("**Compare Checkpoint Parameters**")
            if len(available_eps) >= 2:
                c_a = st.selectbox("Epoch A", options=available_eps, index=0, key="cmp_a_pg")
                c_b = st.selectbox("Epoch B", options=available_eps, index=len(available_eps)-1, key="cmp_b_pg")
                if st.button("⚖️ Calculate Diff", key="btn_cmp_pg"):
                    diff = ckpt_mgr.compare_checkpoints(c_a, c_b)
                    st.json(diff)
    else:
        st.info("No checkpoint snapshots detected in directory.")
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: RUN HISTORY
# ==============================================================================
elif current_page == "6. Run History":
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.markdown('<div class="cr-stat-label">SESSION ARCHIVE · RUNS/RUN_HISTORY.JSON</div>', unsafe_allow_html=True)
    all_runs = session_mgr.list_runs()
    if all_runs:
        st.dataframe(pd.DataFrame(all_runs), use_container_width=True)
    else:
        st.info("No run history sessions recorded yet.")
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: AI TRAINING DOCTOR
# ==============================================================================
elif current_page == "7. AI Training Doctor":
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.markdown('<div class="cr-stat-label">DIAGNOSTIC HYPOTHESIS & ROOT-CAUSE ASSISTANT</div>', unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([3, 1])
    with d_col2:
        use_llm_mode = st.toggle("Enable LLM Mode", value=False)
        if st.button("🔬 Regenerate Diagnosis", key="btn_doc_regen", type="primary"):
            st.rerun()

    doc_report = diagnose_run(
        audit_history=history,
        run_record=selected_run,
        layer_risks=layer_risks,
        use_llm=use_llm_mode,
    )

    badge_lvl = "success" if doc_report.confidence_score >= 80 else "warning"
    st.markdown(f"""
    <div style="margin-top: 1rem; margin-bottom: 1rem;">
        <span style="font-size: 1.35rem; font-weight: 800; color: #22D3EE;">{doc_report.headline}</span>
        <div style="margin-top: 0.35rem;">
            {render_status_badge(doc_report.mode, 'info')}
            {render_status_badge(f"Confidence {doc_report.confidence_score}/100", badge_lvl)}
            {render_status_badge('Rollback Advised' if doc_report.rollback_recommended else 'Proceed', 'critical' if doc_report.rollback_recommended else 'success')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**What Happened:** {doc_report.what_happened}")
    st.markdown(f"**Root Cause Analysis:** {doc_report.likely_root_cause}")

    with st.expander("🔎 Forensic Evidence & Log Observations", expanded=True):
        for ev in doc_report.evidence:
            st.markdown(f"- {ev}")

    st.markdown("### 🛠️ Ranked Remediation Plan")
    for idx, step in enumerate(doc_report.ranked_remediation_steps, 1):
        st.markdown(f"""
        <div class="cr-card" style="margin-bottom: 0.75rem;">
            <div style="font-weight: 700; color: #22D3EE; font-size: 1rem;">#{idx} {step.title} &nbsp; {render_status_badge(step.risk_level, 'warning')}</div>
            <div style="font-size: 0.85rem; color: #E6EDF3; margin-top: 0.35rem;"><strong>Mechanism:</strong> {step.why_it_may_help}</div>
            <div style="font-size: 0.85rem; color: #8B949E; margin-top: 0.2rem;"><strong>Expected Impact:</strong> {step.expected_impact}</div>
        </div>
        """, unsafe_allow_html=True)
        if step.code_snippet:
            st.code(step.code_snippet, language="python")

    st.caption(f"⚠️ {doc_report.disclaimer}")
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 8: FAILURE FORECAST
# ==============================================================================
elif current_page == "8. Failure Forecast":
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.markdown('<div class="cr-stat-label">EARLY WARNING FAILURE FORECASTER</div>', unsafe_allow_html=True)

    fc_c1, fc_c2, fc_c3, fc_c4 = st.columns(4)
    with fc_c1:
        st.markdown(render_stat_box("RISK LEVEL", fc_level.upper(), "Categorical threat", COLORS.get(fc_level.lower(), '#22D3EE')), unsafe_allow_html=True)
    with fc_c2:
        st.markdown(render_stat_box("RISK SCORE", f"{fc_score}/100", "Composite index"), unsafe_allow_html=True)
    with fc_c3:
        st.markdown(render_stat_box("PREDICTED THREAT", fc_threat, "Imminent fault type"), unsafe_allow_html=True)
    with fc_c4:
        est_t = f"~{fc_epochs} epochs" if fc_epochs else "N/A"
        st.markdown(render_stat_box("TIME TO FAILURE", est_t, "Estimated lead time"), unsafe_allow_html=True)

    st.markdown("### ⚠️ Early Precursor Signals")
    for sig in fc_signals:
        st.markdown(f"- {sig}")

    st.markdown("### 🛡️ Recommended Preventive Action")
    st.markdown(f"**{fc_action.get('title', 'Action')}**")
    st.caption(fc_action.get("why_it_may_help", ""))
    if fc_action.get("code_snippet"):
        st.code(fc_action["code_snippet"], language="python")

    # Risk Score Over Time
    st.markdown("### 📈 Risk Evolution Across Epochs")
    epochs_idx = [r.get("epoch", idx) for idx, r in enumerate(history)]
    fc_scores_hist = [r.get("forecast", {}).get("overall_risk_score", 0) if r.get("forecast") else 0 for r in history]
    st.line_chart(pd.DataFrame({"Epoch": epochs_idx, "Risk Score": fc_scores_hist}).set_index("Epoch"), height=240)
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 9: SETTINGS & EXPORTS
# ==============================================================================
elif current_page == "9. Settings":
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.markdown('<div class="cr-stat-label">OBSERVABILITY & EXPORT CONFIGURATION</div>', unsafe_allow_html=True)

    st.markdown("### 📄 Diagnostic Report Exporters")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        if st.button("📥 Markdown Report", use_container_width=True):
            p = export_markdown_report(history, layer_risks, selected_run)
            st.success(f"Saved: {p}")
    with col_e2:
        if st.button("📊 CSV Metrics", use_container_width=True):
            p = export_metrics_csv(history)
            st.success(f"Saved: {p}")
    with col_e3:
        if st.button("⚠️ JSON Faults", use_container_width=True):
            p = export_fault_log_json(history)
            st.success(f"Saved: {p}")
    with col_e4:
        if st.button("🖨️ Printable HTML", use_container_width=True):
            p = export_health_summary_html(history, layer_risks)
            st.success(f"Saved: {p}")

    st.divider()
    st.markdown("### ⚙️ Engine Parameters")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.slider("Fault Sensitivity (Sigma Threshold)", 1.5, 4.0, 2.5, step=0.1)
        st.slider("Checkpoint Retention Count (Latest N)", 2, 20, 5)
    with s_col2:
        st.slider("Rolling History Window Size", 3, 20, 5)
        st.info("🔒 Zero external data transmission unless LLM mode is explicitly activated in Settings.")
    st.markdown('</div>', unsafe_allow_html=True)
