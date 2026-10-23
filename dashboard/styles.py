"""
dashboard/styles.py — Design System tokens, CSS injection, and reusable HTML components
for the MLBlackBox Control Room UI.
"""

# Color Palette Tokens
COLORS = {
    "bg_main": "#0B0F17",
    "bg_card": "#131A24",
    "bg_surface": "#161B22",
    "border": "#263041",
    "border_subtle": "#1E2633",
    "accent_primary": "#22D3EE",
    "accent_cyan": "#22D3EE",
    "accent_blue": "#3B82F6",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "high_risk": "#F97316",
    "critical": "#EF4444",
    "text_main": "#E6EDF3",
    "text_muted": "#8B949E",
}

CONTROL_ROOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: {COLORS['bg_main']} !important;
        color: {COLORS['text_main']} !important;
    }}

    code, pre, .stCode {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Top App Header & Toolbar */
    [data-testid="stHeader"] {{
        background: rgba(11, 15, 23, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid {COLORS['border']} !important;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['bg_card']} !important;
        border-right: 1px solid {COLORS['border']} !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {COLORS['border_subtle']} !important;
        margin: 1rem 0 !important;
    }}

    /* Card Containers */
    .cr-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .cr-card:hover {{
        border-color: #38475c;
    }}

    /* Mini Metric Cards */
    .cr-stat-box {{
        background: {COLORS['bg_surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }}
    .cr-stat-label {{
        color: {COLORS['text_muted']};
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }}
    .cr-stat-value {{
        color: {COLORS['text_main']};
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }}
    .cr-stat-sub {{
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }}

    /* Control Room Hero Header */
    .cr-hero {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 1.25rem 0;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 1.5rem;
    }}
    .cr-hero-title {{
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {COLORS['text_main']};
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    .cr-hero-subtitle {{
        color: {COLORS['text_muted']};
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }}

    /* Status Badges */
    .cr-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .cr-badge-success {{ background: rgba(34, 197, 94, 0.15); color: {COLORS['success']}; border: 1px solid rgba(34, 197, 94, 0.3); }}
    .cr-badge-warning {{ background: rgba(245, 158, 11, 0.15); color: {COLORS['warning']}; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .cr-badge-high {{ background: rgba(249, 115, 22, 0.15); color: {COLORS['high_risk']}; border: 1px solid rgba(249, 115, 22, 0.3); }}
    .cr-badge-critical {{ background: rgba(239, 68, 68, 0.15); color: {COLORS['critical']}; border: 1px solid rgba(239, 68, 68, 0.3); }}
    .cr-badge-info {{ background: rgba(34, 211, 238, 0.15); color: {COLORS['accent_cyan']}; border: 1px solid rgba(34, 211, 238, 0.3); }}

    /* Health Score Dial & Bar */
    .cr-health-container {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.4rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }}
    .cr-health-number {{
        font-size: 2.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }}
    .cr-health-bar-bg {{
        background: #1e2633;
        border-radius: 9999px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin-top: 0.5rem;
    }}
    .cr-health-bar-fill {{
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease;
    }}

    /* Risk Alert Box */
    .cr-alert {{
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }}

    /* Streamlit Widget Overrides */
    div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        color: {COLORS['text_main']} !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {COLORS['text_muted']} !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}

    /* Primary and Secondary Buttons */
    .stButton > button {{
        background-color: {COLORS['bg_surface']} !important;
        color: {COLORS['text_main']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease !important;
    }}
    .stButton > button:hover {{
        background-color: #1f2937 !important;
        border-color: {COLORS['accent_cyan']} !important;
        color: {COLORS['accent_cyan']} !important;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {COLORS['accent_cyan']} !important;
        color: #0B0F17 !important;
        border: 1px solid {COLORS['accent_cyan']} !important;
        font-weight: 700 !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #06b6d4 !important;
        color: #000 !important;
    }}

    /* Custom Radio Navigation */
    div[data-testid="stRadio"] > label {{
        display: none !important;
    }}
    div[data-testid="stRadio"] > div {{
        gap: 0.35rem;
    }}
    div[data-testid="stRadio"] label > div:first-child {{
        display: none !important;
    }}
    div[data-testid="stRadio"] label {{
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.85rem !important;
        color: {COLORS['text_muted']} !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
    }}
    div[data-testid="stRadio"] label:hover {{
        background: {COLORS['bg_surface']} !important;
        color: {COLORS['text_main']} !important;
    }}
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] [aria-checked="true"] {{
        background: rgba(34, 211, 238, 0.1) !important;
        color: {COLORS['accent_cyan']} !important;
        border: 1px solid rgba(34, 211, 238, 0.3) !important;
        font-weight: 600 !important;
    }}

    /* Code Blocks */
    .stCodeBlock {{
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
    }}
</style>
"""


def render_stat_box(label: str, value: str, subtext: str = "", delta_color: str = "#8B949E") -> str:
    """Returns HTML for a sleek mini metric card."""
    sub_html = f'<div class="cr-stat-sub" style="color: {delta_color};">{subtext}</div>' if subtext else ''
    return f"""
    <div class="cr-stat-box">
        <div class="cr-stat-label">{label}</div>
        <div class="cr-stat-value">{value}</div>
        {sub_html}
    </div>
    """


def render_status_badge(text: str, level: str = "info") -> str:
    """Returns HTML for a pill status badge (success, warning, high, critical, info)."""
    return f'<span class="cr-badge cr-badge-{level}">{text}</span>'


def render_alert_box(title: str, message: str, level: str = "warning") -> str:
    """Returns HTML for a full-width alert box."""
    colors = {
        "success": (COLORS["success"], "rgba(34, 197, 94, 0.1)", "rgba(34, 197, 94, 0.3)"),
        "warning": (COLORS["warning"], "rgba(245, 158, 11, 0.1)", "rgba(245, 158, 11, 0.3)"),
        "high": (COLORS["high_risk"], "rgba(249, 115, 22, 0.1)", "rgba(249, 115, 22, 0.3)"),
        "critical": (COLORS["critical"], "rgba(239, 68, 68, 0.1)", "rgba(239, 68, 68, 0.3)"),
        "info": (COLORS["accent_cyan"], "rgba(34, 211, 238, 0.1)", "rgba(34, 211, 238, 0.3)"),
    }
    fg, bg, bd = colors.get(level, colors["info"])
    return f"""
    <div class="cr-alert" style="background: {bg}; border: 1px solid {bd};">
        <div style="font-size: 1.25rem; color: {fg};">●</div>
        <div>
            <div style="font-weight: 700; color: {fg}; font-size: 0.95rem;">{title}</div>
            <div style="color: {COLORS['text_muted']}; font-size: 0.85rem; margin-top: 0.2rem;">{message}</div>
        </div>
    </div>
    """


def render_health_card(score: int, explanation: str) -> str:
    """Returns HTML for the large Health Score card with progress bar."""
    if score >= 80:
        col = COLORS["success"]
    elif score >= 50:
        col = COLORS["warning"]
    else:
        col = COLORS["critical"]

    return f"""
    <div class="cr-card" style="display: flex; align-items: center; gap: 1.5rem; height: 100%;">
        <div style="text-align: center; min-width: 110px;">
            <div class="cr-stat-label" style="margin-bottom: 0;">HEALTH SCORE</div>
            <div class="cr-health-number" style="color: {col};">{score}</div>
            <div style="font-size: 0.75rem; color: {COLORS['text_muted']}; font-weight: 600;">/ 100</div>
        </div>
        <div style="flex: 1;">
            <div style="font-weight: 700; font-size: 1.05rem; color: {COLORS['text_main']};">Training Convergence Status</div>
            <div style="font-size: 0.85rem; color: {COLORS['text_muted']}; margin-top: 0.25rem; line-height: 1.4;">{explanation}</div>
            <div class="cr-health-bar-bg">
                <div class="cr-health-bar-fill" style="width: {score}%; background: {col};"></div>
            </div>
        </div>
    </div>
    """


def render_forecast_banner(level: str, score: int, threat: str, epochs_left: int = None, signal: str = "") -> str:
    """Returns HTML for the Control Room forecast badge card."""
    colors = {
        "Low": (COLORS["success"], "rgba(34, 197, 94, 0.1)", "rgba(34, 197, 94, 0.25)"),
        "Medium": (COLORS["warning"], "rgba(245, 158, 11, 0.1)", "rgba(245, 158, 11, 0.25)"),
        "High": (COLORS["high_risk"], "rgba(249, 115, 22, 0.1)", "rgba(249, 115, 22, 0.25)"),
        "Critical": (COLORS["critical"], "rgba(239, 68, 68, 0.1)", "rgba(239, 68, 68, 0.25)"),
    }
    fg, bg, bd = colors.get(level, colors["Low"])
    est_str = f"In ~{epochs_left} epochs" if epochs_left else "No imminent divergence"

    return f"""
    <div class="cr-card" style="background: {bg}; border: 1px solid {bd}; height: 100%; display: flex; align-items: center; gap: 1.5rem;">
        <div style="text-align: center; min-width: 100px;">
            <div class="cr-stat-label" style="color: {fg}; margin-bottom: 0;">FORECAST RISK</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: {fg}; font-family: 'JetBrains Mono', monospace;">{level.upper()}</div>
            <div style="font-size: 0.75rem; color: {fg}; font-weight: 600;">{score}/100</div>
        </div>
        <div style="flex: 1;">
            <div style="font-weight: 700; font-size: 1.05rem; color: {COLORS['text_main']};">Threat: {threat}</div>
            <div style="font-size: 0.85rem; color: {COLORS['text_muted']}; margin-top: 0.2rem;">{est_str}</div>
            <div style="font-size: 0.8rem; color: {fg}; margin-top: 0.35rem; font-style: italic;">{signal or 'Nominal gradient dynamics'}</div>
        </div>
    </div>
    """
