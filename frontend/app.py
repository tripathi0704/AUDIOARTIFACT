"""
app.py — AudioArtifact Enterprise Forensic Dashboard
---------------------------------------------------------
Comprehensive Timeline-Based Deepfake Audio Localizer & Forensic Suite.

Features:
- Tab 1: Single Audio Forensic Localizer (Verdict Cards, Probability Timeline,
         3D Voice Signature Spectrogram, Segment-by-Segment Click-to-Play Audio Slicer,
         Cryptographic SHA-256 Hash Audit, Biological & Acoustic Artifacts,
         Adaptive Sensitivity Threshold Slider, 1-Click Forensic HTML Report Download)
- Tab 2: Live Microphone Voice Test (Real-time browser mic recording & instant analysis)
- Tab 3: Speaker Clone & Voiceprint Matcher (Reference vs Suspect WavLM 768-dim cross-matching)
- Tab 4: Batch Forensic Scanner (Multi-file bulk upload, progress tracking, CSV export)
- Tab 5: Session History & Audit Log (Search, verdict filters, CSV export, one-click re-inspection)
"""

import os
import sys
import types
import io
import json
import uuid
from datetime import datetime, timezone, timedelta
import zoneinfo
import requests
import pandas as pd
import streamlit.components.v1 as components

# Windows 11 Smart App Control & cross-platform safe numba bypass for librosa
try:
    from numba import _dispatcher
except Exception:
    import importlib.machinery
    _m = types.ModuleType("numba")
    _m.__spec__ = importlib.machinery.ModuleSpec("numba", None)
    _m.jit = lambda *a, **kw: (lambda f: f) if a and callable(a[0]) else (lambda f: f)
    _m.njit = _m.jit
    _m.vectorize = _m.jit
    _m.guvectorize = _m.jit
    _m.stencil = _m.jit
    _m.prange = range
    sys.modules["numba"] = _m
    _m_core = types.ModuleType("numba.core")
    _m_core.__spec__ = importlib.machinery.ModuleSpec("numba.core", None)
    sys.modules["numba.core"] = _m_core
    sys.modules["numba.core.decorators"] = _m

# Prevent torchaudio DLL loading failures on Windows (Smart App Control / C++ extension issues)
# torchaudio is not used by AudioArtifact (we use soundfile & librosa)
if "torchaudio" not in sys.modules:
    import importlib.machinery
    # Windows 11 Smart App Control & Code Integrity safe dummy mock with valid ModuleSpec
    _m_ta = types.ModuleType("torchaudio")
    _m_ta.__version__ = "2.14.0"
    _m_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = _m_ta
    _m_ta_lib = types.ModuleType("torchaudio.lib")
    _m_ta_lib.__spec__ = importlib.machinery.ModuleSpec("torchaudio.lib", None)
    sys.modules["torchaudio.lib"] = _m_ta_lib
    _m_ta_c = types.ModuleType("torchaudio.lib._torchaudio")
    _m_ta_c.__spec__ = importlib.machinery.ModuleSpec("torchaudio.lib._torchaudio", None)
    sys.modules["torchaudio.lib._torchaudio"] = _m_ta_c

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import librosa

# Support importing local backend modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.main import (
    analyze_audio_data,
    extract_speaker_embedding_vector,
    decode_audio_bytes,
    get_model,
    get_wavlm,
    set_wavlm,
)
from backend.forensic_utils import (
    compute_speaker_similarity,
    generate_forensic_html_report,
    slice_segment_audio_bytes,
    compute_file_hashes,
    compute_acoustic_forensics,
    build_police_complaint_pdf,
)
from backend.ai_copilot import (
    generate_forensic_explanation,
    assess_scam_threat,
    chat_with_audio_copilot,
    is_genai_installed,
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def get_active_gemini_api_key() -> str:
    """Resolves Gemini API key with priority: UI input > st.secrets > os.environ > .env."""
    user_key = st.session_state.get("custom_gemini_api_key", "").strip()
    if user_key:
        return user_key
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


def is_local_backend_active(port: int = 8000) -> bool:
    """Non-blocking check to see if local uvicorn is listening on port 8000."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.05)
        s.connect(("127.0.0.1", port))
        s.close()
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner="Initializing forensic foundation models (WavLM & Detector)...")
def preload_cloud_models():
    """Cache foundation models in memory on startup."""
    from backend.main import get_model, get_wavlm, set_wavlm
    m = get_model()
    ext, wavlm = get_wavlm()
    set_wavlm(ext, wavlm)
    return m, ext, wavlm


# If local backend on port 8000 is not active, preload models once in Streamlit process
if not is_local_backend_active(8000):
    _cached_clf, _cached_ext, _cached_wavlm = preload_cloud_models()
    from backend.main import set_wavlm
    set_wavlm(_cached_ext, _cached_wavlm)

st.set_page_config(
    page_title="AudioArtifact — Forensic Audio Suite",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ephemeral session initialization
if "session_token" not in st.session_state:
    st.session_state["session_token"] = uuid.uuid4().hex[:6].upper()
if "session_scans" not in st.session_state:
    st.session_state["session_scans"] = []
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = []
if "user_tz_override" not in st.session_state:
    st.session_state["user_tz_override"] = "AUTO"


# ----------------------------------------------------------------------
# CLIENT TIMEZONE & LOCAL TIME SYSTEM
# ----------------------------------------------------------------------
COMMON_TIMEZONES = [
    ("🌐 Auto-Detect Browser Timezone", "AUTO"),
    ("🇮🇳 India (IST, UTC+05:30) — Asia/Kolkata", "Asia/Kolkata"),
    ("🇺🇸 US Eastern (EST/EDT) — America/New_York", "America/New_York"),
    ("🇺🇸 US Central (CST/CDT) — America/Chicago", "America/Chicago"),
    ("🇺🇸 US Mountain (MST/MDT) — America/Denver", "America/Denver"),
    ("🇺🇸 US Pacific (PST/PDT) — America/Los_Angeles", "America/Los_Angeles"),
    ("🇬🇧 United Kingdom (GMT/BST) — Europe/London", "Europe/London"),
    ("🇪🇺 Central Europe (CET/CEST) — Europe/Paris", "Europe/Paris"),
    ("🇦🇪 UAE (GST, UTC+04:00) — Asia/Dubai", "Asia/Dubai"),
    ("🇸🇬 Singapore / Malaysia (SGT, UTC+08:00) — Asia/Singapore", "Asia/Singapore"),
    ("🇯🇵 Japan (JST, UTC+09:00) — Asia/Tokyo", "Asia/Tokyo"),
    ("🇦🇺 Australia Eastern (AEST/AEDT) — Australia/Sydney", "Australia/Sydney"),
    ("🌍 Universal Coordinated Time — UTC", "UTC"),
]


def resolve_client_timezone():
    """
    Detects the visitor's local country timezone dynamically.
    Priority order:
    1. User manual override (if set in session_state and != 'AUTO')
    2. Query param ?tz= (e.g. ?tz=Asia/Kolkata)
    3. Streamlit native browser context st.context.timezone (IANA string, e.g. 'Asia/Kolkata')
    4. Streamlit native timezone offset st.context.timezone_offset (minutes from UTC)
    5. Fallback to host local machine timezone (for localhost development)
    6. Default to 'Asia/Kolkata' (IST) or UTC.
    Returns: (tz_obj, tz_name, tz_display_label, is_auto)
    """
    selected_choice = st.session_state.get("user_tz_override", "AUTO")

    # If manual specific timezone selected
    if selected_choice and selected_choice != "AUTO":
        try:
            tz_obj = zoneinfo.ZoneInfo(selected_choice)
            now_sample = datetime.now(timezone.utc).astimezone(tz_obj)
            abbr = now_sample.strftime("%Z")
            lbl = f"{selected_choice} ({abbr})" if abbr and not abbr.startswith("+") and not abbr.startswith("-") else selected_choice
            return tz_obj, selected_choice, lbl, False
        except Exception:
            pass

    # 1. Check Query Parameter (?tz=)
    query_tz = st.query_params.get("tz")
    if query_tz:
        try:
            tz_obj = zoneinfo.ZoneInfo(query_tz)
            now_sample = datetime.now(timezone.utc).astimezone(tz_obj)
            abbr = now_sample.strftime("%Z")
            lbl = f"{query_tz} ({abbr})" if abbr and not abbr.startswith("+") and not abbr.startswith("-") else query_tz
            return tz_obj, query_tz, lbl, True
        except Exception:
            pass

    # 2. Check Streamlit 1.61+ Browser Context Timezone
    ctx_tz = getattr(st.context, "timezone", None)
    if ctx_tz:
        try:
            tz_obj = zoneinfo.ZoneInfo(ctx_tz)
            now_sample = datetime.now(timezone.utc).astimezone(tz_obj)
            abbr = now_sample.strftime("%Z")
            lbl = f"{ctx_tz} ({abbr})" if abbr and not abbr.startswith("+") and not abbr.startswith("-") else ctx_tz
            return tz_obj, ctx_tz, lbl, True
        except Exception:
            pass

    # 3. Check Streamlit Timezone Offset (minutes)
    ctx_offset = getattr(st.context, "timezone_offset", None)
    if ctx_offset is not None:
        try:
            tz_obj = timezone(-timedelta(minutes=ctx_offset))
            if ctx_offset == -330:
                return tz_obj, "Asia/Kolkata", "Asia/Kolkata (IST)", True
            elif ctx_offset in (240, 300):
                return tz_obj, "America/New_York", "US Eastern (EDT/EST)", True
            elif ctx_offset in (420, 480):
                return tz_obj, "America/Los_Angeles", "US Pacific (PDT/PST)", True
            elif ctx_offset in (0, -60):
                return tz_obj, "Europe/London", "Europe/London (GMT/BST)", True
            else:
                hrs = -ctx_offset // 60
                mins = abs(-ctx_offset % 60)
                sign = "+" if hrs >= 0 else "-"
                lbl = f"UTC{sign}{abs(hrs):02d}:{mins:02d}"
                return tz_obj, lbl, lbl, True
        except Exception:
            pass

    # 4. Fallback to local machine timezone (important for localhost)
    try:
        local_sys_tz = datetime.now().astimezone().tzinfo
        if local_sys_tz and str(local_sys_tz) != "UTC":
            sys_name = getattr(local_sys_tz, "key", str(local_sys_tz))
            now_sample = datetime.now(timezone.utc).astimezone(local_sys_tz)
            abbr = now_sample.strftime("%Z")
            lbl = f"{sys_name} ({abbr})" if abbr else str(sys_name)
            return local_sys_tz, sys_name, lbl, True
    except Exception:
        pass

    # 5. Default fallback to Asia/Kolkata (IST)
    try:
        ist_tz = zoneinfo.ZoneInfo("Asia/Kolkata")
        return ist_tz, "Asia/Kolkata", "Asia/Kolkata (IST)", True
    except Exception:
        return timezone.utc, "UTC", "UTC", True


def format_local_timestamp(dt_or_str=None, tz_obj=None, tz_name="") -> str:
    """
    Formats a datetime or ISO UTC string into a clean local timestamp string.
    Example: 2026-09-14 08:05:22 PM IST
    """
    if dt_or_str is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(dt_or_str, str):
        try:
            clean_str = dt_or_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            return dt_or_str
    elif isinstance(dt_or_str, datetime):
        dt = dt_or_str
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        return str(dt_or_str)

    if tz_obj is not None:
        try:
            dt = dt.astimezone(tz_obj)
        except Exception:
            pass

    abbr = dt.strftime("%Z")
    if not abbr or abbr.startswith("+") or abbr.startswith("-"):
        abbr = tz_name.split("/")[-1] if "/" in tz_name else tz_name
    return dt.strftime("%Y-%m-%d %I:%M:%S %p") + (f" {abbr}" if abbr else "")

# ----------------------------------------------------------------------
# FORENSIC VISUALIZATION HELPERS (PROTOTYPE DESIGN SYSTEM)
# ----------------------------------------------------------------------
def build_holographic_gauge(fake_ratio: float, verdict: str) -> go.Figure:
    """Builds a futuristic circular holographic confidence dial."""
    ai_pct = round(float(fake_ratio), 1)
    if ai_pct < 25:
        bar_color = "#10B981"  # Emerald Green
        verdict_badge = "AUTHENTIC HUMAN"
    elif ai_pct < 65:
        bar_color = "#F59E0B"  # Amber Alert
        verdict_badge = "SUSPICIOUS / SPLICED"
    else:
        bar_color = "#FF4B72"  # Coral / Crimson
        verdict_badge = "SYNTHETIC DEEPFAKE"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ai_pct,
        number={
            "suffix": "%",
            "font": {"size": 40, "family": "Outfit, Space Grotesk, sans-serif", "color": bar_color}
        },
        title={
            "text": f"<b>SYNTHETIC RISK</b><br><span style='font-size:10.5px;color:#94A3B8;letter-spacing:0.12em;font-family:JetBrains Mono;'>{verdict_badge}</span>",
            "font": {"size": 13, "family": "Outfit, sans-serif", "color": "#F8FAFC"}
        },
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "tickfont": {"size": 9, "color": "#64748B"}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(13, 18, 30, 0.6)",
            "borderwidth": 1,
            "bordercolor": "rgba(0, 245, 255, 0.18)",
            "steps": [
                {"range": [0, 25], "color": "rgba(16, 185, 129, 0.14)"},
                {"range": [25, 65], "color": "rgba(245, 158, 11, 0.14)"},
                {"range": [65, 100], "color": "rgba(255, 75, 114, 0.16)"}
            ]
        }
    ))
    fig.update_layout(
        height=225,
        margin=dict(l=15, r=15, t=35, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC")
    )
    return fig


def build_anomaly_radar_chart(forensics: dict, fake_ratio: float) -> go.Figure:
    """Builds an executive forensic acoustic radar chart comparing target audio vs human baseline."""
    categories = [
        "Spectral Rolloff",
        "Pitch Inflection",
        "ZCR Dynamics",
        "Dynamic Range",
        "Harmonic Energy"
    ]

    rolloff = forensics.get("spectral_rolloff_85_hz", 2400) if forensics else 2400
    rolloff_score = min(100, max(15, int((rolloff / 4800.0) * 100)))

    pitch_score = min(100, max(10, int(forensics.get("pitch_inflection_score", 45)))) if forensics else 45
    zcr = forensics.get("zero_crossing_rate", 0.08) if forensics else 0.08
    zcr_score = min(100, max(15, int(zcr * 500)))

    dyn = forensics.get("dynamic_range_db", 35.0) if forensics else 35.0
    dyn_score = min(100, max(15, int(100 - min(80, dyn * 1.8))))

    harm_score = min(100, max(20, int(float(fake_ratio) * 0.85 + 15)))

    suspect_values = [rolloff_score, pitch_score, zcr_score, dyn_score, harm_score]
    human_baseline = [32, 75, 30, 35, 25]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=human_baseline + [human_baseline[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.08)",
        line=dict(color="rgba(16, 185, 129, 0.6)", width=1.5, dash="dot"),
        name="Human Baseline",
        hoverinfo="skip"
    ))

    line_color = "#FF4B72" if fake_ratio >= 50 else "#00F5FF"
    fill_color = "rgba(255, 75, 114, 0.22)" if fake_ratio >= 50 else "rgba(0, 245, 255, 0.18)"

    fig.add_trace(go.Scatterpolar(
        r=suspect_values + [suspect_values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2.5),
        name="Analyzed Audio",
        marker=dict(size=5, color=line_color)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor="rgba(255,255,255,0.08)", gridcolor="rgba(255,255,255,0.07)"),
            angularaxis=dict(tickfont=dict(size=9.5, family="Outfit, sans-serif", color="#94A3B8"), linecolor="rgba(255,255,255,0.08)", gridcolor="rgba(255,255,255,0.07)"),
            bgcolor="rgba(13, 18, 30, 0.45)"
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10, family="JetBrains Mono", color="#94A3B8")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=225,
        margin=dict(l=25, r=25, t=15, b=25)
    )
    return fig


def build_waveform_and_timeline_chart(segments: list, total_dur: float, spliced_regions: list, decoded_y, threshold_pct: float) -> go.Figure:
    """Builds a synchronized dual-track timeline: Audio Waveform with Spliced Highlights + Deepfake Probability Curve."""
    has_audio_track = decoded_y is not None and len(decoded_y) > 0

    if has_audio_track:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.09,
            row_heights=[0.42, 0.58],
            subplot_titles=("ACOUSTIC WAVEFORM & SPLICED ANOMALY REGIONS", "CONTINUOUS DEEPFAKE PROBABILITY TIMELINE")
        )

        y_arr = np.asarray(decoded_y, dtype=np.float32)
        target_pts = 1200
        step = max(1, len(y_arr) // target_pts)
        y_down = y_arr[::step]
        t_down = np.linspace(0, total_dur, len(y_down))

        fig.add_trace(go.Scatter(
            x=t_down,
            y=y_down,
            mode="lines",
            line=dict(color="#00F5FF", width=1.15),
            name="Audio Envelope",
            hoverinfo="skip"
        ), row=1, col=1)

        for s_idx, r in enumerate(spliced_regions or []):
            fig.add_vrect(
                x0=r.get("start", 0),
                x1=r.get("end", 0),
                fillcolor="rgba(245, 158, 11, 0.28)",
                line=dict(color="#F59E0B", width=1.5),
                annotation_text="🎯 SPLICED AI",
                annotation_position="top left",
                annotation_font=dict(size=9.5, color="#F59E0B", family="JetBrains Mono"),
                row=1, col=1
            )
    else:
        fig = go.Figure()

    time_points = [round((s["start"] + s["end"]) / 2, 2) for s in segments]
    fake_probs = [round(s["fake_probability"] * 100, 2) for s in segments]
    hover_texts = [
        f"<b>Window #{s.get('segment', idx) + 1}</b><br>"
        f"Timeline: <b>{s.get('start', 0)}s – {s.get('end', 0)}s</b><br>"
        f"Synthetic Risk: <b>{s.get('fake_probability', 0)*100:.1f}%</b><br>"
        f"Status: <b>{s.get('label', 'N/A')}</b> (Conf: {s.get('confidence', 0)}%)"
        for idx, s in enumerate(segments)
    ]

    prob_trace = go.Scatter(
        x=time_points,
        y=fake_probs,
        mode="lines+markers",
        name="Deepfake Probability",
        line=dict(color="#10B981", width=3, shape="spline"),
        marker=dict(
            size=7,
            color=["#FF4B72" if p >= threshold_pct else "#10B981" for p in fake_probs],
            line=dict(color="#07090E", width=1.5)
        ),
        hovertext=hover_texts,
        hoverinfo="text"
    )

    thresh_trace = go.Scatter(
        x=[0, total_dur],
        y=[threshold_pct, threshold_pct],
        mode="lines",
        line=dict(color="rgba(245, 158, 11, 0.8)", width=1.5, dash="dash"),
        name=f"Decision Boundary ({threshold_pct}%)",
        hoverinfo="skip"
    )

    if has_audio_track:
        fig.add_trace(thresh_trace, row=2, col=1)
        fig.add_trace(prob_trace, row=2, col=1)
        fig.add_shape(
            type="rect",
            x0=0, x1=total_dur,
            y0=threshold_pct, y1=100,
            fillcolor="rgba(255, 75, 114, 0.06)",
            line=dict(width=0),
            layer="below",
            row=2, col=1
        )
    else:
        fig.add_trace(thresh_trace)
        fig.add_trace(prob_trace)
        fig.add_shape(
            type="rect",
            x0=0, x1=total_dur,
            y0=threshold_pct, y1=100,
            fillcolor="rgba(255, 75, 114, 0.06)",
            line=dict(width=0),
            layer="below"
        )

    layout_args = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0D121E",
        height=430 if has_audio_track else 320,
        margin=dict(l=35, r=20, t=35 if has_audio_track else 15, b=35),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(family="JetBrains Mono", size=10, color="#94A3B8")
        )
    )

    if has_audio_track:
        fig.update_layout(**layout_args)
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False, range=[0, total_dur], row=1, col=1)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False, row=1, col=1)
        fig.update_xaxes(title=dict(text="Audio Timeline (Seconds)", font=dict(family="JetBrains Mono", size=11, color="#94A3B8")), gridcolor="rgba(255,255,255,0.06)", zeroline=False, range=[0, total_dur], row=2, col=1)
        fig.update_yaxes(title=dict(text="Synthetic Risk (%)", font=dict(family="JetBrains Mono", size=11, color="#94A3B8")), gridcolor="rgba(255,255,255,0.06)", range=[-3, 105], zeroline=False, row=2, col=1)
        for ann in fig["layout"]["annotations"]:
            ann["font"] = dict(family="Outfit, sans-serif", size=11.5, color="#F8FAFC")
    else:
        fig.update_layout(
            **layout_args,
            xaxis=dict(
                title=dict(text="Audio Timeline (Seconds)", font=dict(family="JetBrains Mono", size=11, color="#94A3B8")),
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                range=[0, total_dur]
            ),
            yaxis=dict(
                title=dict(text="Synthetic Risk (%)", font=dict(family="JetBrains Mono", size=11, color="#94A3B8")),
                gridcolor="rgba(255,255,255,0.06)",
                range=[-3, 105],
                zeroline=False
            )
        )
    return fig


# ----------------------------------------------------------------------
# GLOBAL CYBER-FORENSIC GLASSMORPHISM STYLING
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg: #07090E;
  --panel: rgba(13, 18, 30, 0.72);
  --panel-solid: #0D121E;
  --panel-hover: rgba(18, 25, 42, 0.85);
  --line: rgba(0, 245, 255, 0.14);
  --line-subtle: rgba(255, 255, 255, 0.08);
  --text: #F8FAFC;
  --muted: #94A3B8;
  --muted-dim: #64748B;
  --cyan: #00F5FF;
  --cyan-glow: rgba(0, 245, 255, 0.35);
  --emerald: #10B981;
  --emerald-dim: rgba(16, 185, 129, 0.12);
  --coral: #FF4B72;
  --coral-dim: rgba(255, 75, 114, 0.14);
  --amber: #F59E0B;
  --amber-dim: rgba(245, 158, 11, 0.14);
  --violet: #8B5CF6;
}

#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
  visibility: hidden;
  height: 0;
}

.stApp {
  background-color: var(--bg);
  background-image: 
    radial-gradient(circle at 50% 0%, rgba(0, 245, 255, 0.06) 0%, transparent 50%),
    radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.04) 0%, transparent 40%),
    radial-gradient(circle at 15% 70%, rgba(16, 185, 129, 0.04) 0%, transparent 40%);
  background-attachment: fixed;
}

.block-container {
  padding-top: 1.6rem;
  padding-bottom: 3.5rem;
  max-width: 1260px;
}

html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif;
  color: var(--text);
}

.mono {
  font-family: 'JetBrains Mono', monospace;
}

/* Mission Command Navigation Bar */
.brand-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 22px;
  padding: 12px 18px;
  background: var(--panel);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
}

.brand {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 800;
  font-size: 19px;
  display: flex;
  align-items: center;
  gap: 12px;
  letter-spacing: -0.01em;
  color: #FFFFFF;
}

.brand-orb {
  width: 11px;
  height: 11px;
  background: var(--cyan);
  border-radius: 50%;
  box-shadow: 0 0 14px var(--cyan);
  animation: pulse-glow 2.4s infinite;
}

@keyframes pulse-glow {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.45; transform: scale(0.88); }
}

.status-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  background: rgba(13, 18, 30, 0.85);
  border: 1px solid var(--line-subtle);
  padding: 5px 12px;
  border-radius: 20px;
  display: flex;
  align-items: center;
  gap: 7px;
}

.hero-banner {
  margin-bottom: 24px;
}

.hero-title {
  font-family: 'Outfit', sans-serif;
  font-weight: 900;
  letter-spacing: -0.03em;
  font-size: clamp(30px, 4.2vw, 46px);
  line-height: 1.12;
  margin-bottom: 6px;
  color: #FFFFFF;
}

.hero-title span.cyan {
  color: var(--cyan);
  text-shadow: 0 0 24px rgba(0, 245, 255, 0.35);
}

.hero-title span.coral {
  color: var(--coral);
  text-shadow: 0 0 24px rgba(255, 75, 114, 0.35);
}

.hero-sub {
  color: var(--muted);
  font-size: 14.5px;
  max-width: 820px;
  line-height: 1.6;
}

/* Tabs Restyling */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: var(--panel);
  backdrop-filter: blur(14px);
  padding: 6px 10px;
  border-radius: 14px;
  border: 1px solid var(--line);
  margin-bottom: 24px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}

.stTabs [data-baseweb="tab"] {
  font-family: 'Outfit', sans-serif;
  font-weight: 600;
  font-size: 14px;
  color: var(--muted);
  border-radius: 10px;
  padding: 8px 18px;
  border: none !important;
  background-color: transparent !important;
  transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
  background: rgba(0, 245, 255, 0.12) !important;
  color: var(--cyan) !important;
  border: 1px solid rgba(0, 245, 255, 0.35) !important;
  box-shadow: 0 4px 16px rgba(0, 245, 255, 0.15);
}

/* File Uploader Restyle */
[data-testid="stFileUploader"] {
  background: var(--panel);
  backdrop-filter: blur(14px);
  border: 1.5px dashed rgba(0, 245, 255, 0.25);
  border-radius: 16px;
  padding: 12px 16px;
  transition: all 0.25s ease;
}

[data-testid="stFileUploader"]:hover {
  border-color: var(--cyan);
  box-shadow: 0 0 20px rgba(0, 245, 255, 0.15);
}

/* Buttons */
div.stButton > button {
  background: linear-gradient(135deg, #00F5FF 0%, #00C4D4 100%);
  color: #07090E;
  border: none;
  font-weight: 700;
  font-family: 'Outfit', sans-serif;
  font-size: 14px;
  padding: 10px 24px;
  border-radius: 24px;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  box-shadow: 0 4px 16px rgba(0, 245, 255, 0.25);
}

div.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 245, 255, 0.4);
}

/* Audio Player */
audio {
  filter: invert(0.92) hue-rotate(180deg) brightness(1.1);
  border-radius: 30px;
  width: 100%;
}

/* Section Labels */
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11.5px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--cyan);
  margin: 30px 0 14px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.section-label::before {
  content: '';
  width: 20px;
  height: 2px;
  background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan);
}

/* Cyber Cards */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin-bottom: 22px;
}

.vcard {
  background: var(--panel);
  backdrop-filter: blur(14px);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px 20px;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4);
}

.vcard:hover {
  transform: translateY(-2px);
  border-color: rgba(0, 245, 255, 0.35);
  box-shadow: 0 12px 36px rgba(0, 245, 255, 0.1);
}

.vcard .k {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  color: var(--muted-dim);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 6px;
}

.vcard .v {
  font-size: 26px;
  font-weight: 800;
  font-family: 'Outfit', sans-serif;
}

.vcard .sub {
  font-size: 12px;
  color: var(--muted);
  margin-top: 4px;
}

.pill-status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 16px;
  border-radius: 20px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.pill-status.real {
  background: var(--emerald-dim);
  border: 1px solid rgba(16, 185, 129, 0.4);
  color: var(--emerald);
  box-shadow: 0 0 16px rgba(16, 185, 129, 0.15);
}

.pill-status.fake {
  background: var(--coral-dim);
  border: 1px solid rgba(255, 75, 114, 0.4);
  color: var(--coral);
  box-shadow: 0 0 16px rgba(255, 75, 114, 0.15);
}

.pill-status.mixed {
  background: var(--amber-dim);
  border: 1px solid rgba(245, 158, 11, 0.4);
  color: var(--amber);
  box-shadow: 0 0 16px rgba(245, 158, 11, 0.15);
}

.crypto-strip {
  background: var(--panel);
  backdrop-filter: blur(14px);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 18px;
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11.5px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# BRAND MISSION COMMAND HEADER
# ----------------------------------------------------------------------
user_tz_obj, user_tz_name, user_tz_label, is_auto_tz = resolve_client_timezone()
now_local = datetime.now(timezone.utc).astimezone(user_tz_obj)
initial_clock_str = now_local.strftime("%I:%M:%S %p")
backend_online = is_local_backend_active(8000)

st.markdown(f"""
<div class="brand-nav">
  <div class="brand">
    <span class="brand-orb"></span>
    <span>Audio<span style="color:var(--cyan);">Artifact</span> <span style="font-size:12px;color:var(--muted);font-weight:500;margin-left:4px;">FORENSIC LAB</span></span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <div class="status-badge" style="color:var(--cyan);border-color:rgba(0,245,255,0.35);">
      ◈ SESSION #{st.session_state['session_token']}
    </div>
    <div class="status-badge" style="color:#FFFFFF;border-color:rgba(0,245,255,0.25);">
      <span style="color:var(--cyan);">🕒</span>
      <span style="font-weight:700;">{initial_clock_str}</span>
      <span style="color:var(--muted);font-size:10.5px;margin-left:4px;">· {user_tz_label}</span>
    </div>
    <div class="status-badge" style="color:{'#10B981' if backend_online else '#00F5FF'};border-color:{'rgba(16,185,129,0.35)' if backend_online else 'rgba(0,245,255,0.35)'};">
      <span style="color:{'#10B981' if backend_online else '#00F5FF'};">●</span>
      <span>{'FASTAPI BACKEND ACTIVE' if backend_online else 'NEURAL IN-MEMORY PIPELINE'}</span>
    </div>
  </div>
</div>
<div class="hero-banner">
  <h1 class="hero-title">Audio<span class="cyan">Artifact</span> — Forensic <span class="coral">Deepfake</span> Intelligence</h1>
  <p class="hero-sub">
    Next-generation neural acoustic forensics: continuous temporal localization, acoustic bio-marker verification,
    voiceprint clone cross-matching, and court-ready cryptographic audit certification.
  </p>
</div>
""", unsafe_allow_html=True)



# ----------------------------------------------------------------------
# SIDEBAR CONFIGURATION (AI COPILOT & CLIENT PREFERENCES)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-weight:700;font-size:14px;color:#ECEFEB;margin-bottom:12px;">◈ FORENSIC ENGINE CONFIG</div>', unsafe_allow_html=True)
    
    with st.expander("🤖 Gemini AI Copilot Settings", expanded=True):
        active_gemini_key = get_active_gemini_api_key()
        if active_gemini_key:
            st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11.5px;color:#3ECF8E;background:rgba(62,207,142,0.1);padding:5px 10px;border-radius:6px;border:1px solid rgba(62,207,142,0.25);margin-bottom:10px;">🟢 Gemini Ultra-Fast Flash Engine Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11.5px;color:#FFB454;background:rgba(255,180,84,0.1);padding:5px 10px;border-radius:6px;border:1px solid rgba(255,180,84,0.25);margin-bottom:10px;">⚪ API Key Optional / Not Set</div>', unsafe_allow_html=True)

        user_input_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("custom_gemini_api_key", ""),
            type="password",
            placeholder="Paste AI Studio API Key...",
            help="Required for Explainable AI (XAI), Threat Intent Analysis, and Copilot Chat. Free at aistudio.google.com",
            key="input_custom_gemini_key",
        )
        if user_input_key != st.session_state.get("custom_gemini_api_key", ""):
            st.session_state["custom_gemini_api_key"] = user_input_key
            st.rerun()

        st.markdown(
            """<div style="font-size:11px;color:#8C958E;margin-top:8px;line-height:1.4;">
            Get a free API key at <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#3ECF8E;text-decoration:none;">Google AI Studio</a>.<br/>
            On Streamlit Cloud, you can also add <code>GEMINI_API_KEY</code> in Secrets.
            </div>""",
            unsafe_allow_html=True
        )

# ----------------------------------------------------------------------
# TOP NAVIGATION TABS & REDIRECTION CONTROLLER
# ----------------------------------------------------------------------
TAB_TITLES = [
    "◈ Single Audio Localizer",
    "🎙️ Live Mic Voice Test",
    "👥 Voiceprint Clone Matcher",
    "📁 Batch Forensic Scanner",
    "📜 Session History & Audit Log",
]


def reinspect_scan_callback(record_result: dict, record_filename: str, record_audio_bytes: bytes = None):
    """Updates session state and redirects active tab to Single Audio Localizer."""
    st.session_state["last_result"] = record_result
    st.session_state["analyzed_file_name"] = record_filename
    st.session_state["analyzed_audio_bytes"] = record_audio_bytes
    st.session_state["main_tabs"] = TAB_TITLES[0]
    st.session_state["redirect_tab"] = TAB_TITLES[0]


if "redirect_tab" in st.session_state:
    st.session_state["main_tabs"] = st.session_state.pop("redirect_tab")
elif "main_tabs" not in st.session_state:
    st.session_state["main_tabs"] = TAB_TITLES[0]

tab_single, tab_mic, tab_speaker, tab_batch, tab_history = st.tabs(
    TAB_TITLES,
    key="main_tabs",
    on_change="rerun",
)


def execute_forensic_scan(file_bytes: bytes, filename: str) -> dict:
    """Executes forensic scan using FastAPI backend or direct in-memory fallback."""
    data = None
    use_http = False
    if BACKEND_URL and ("127.0.0.1:8000" in BACKEND_URL or "localhost:8000" in BACKEND_URL):
        use_http = is_local_backend_active(8000)
    elif BACKEND_URL and not ("127.0.0.1" in BACKEND_URL or "localhost" in BACKEND_URL):
        use_http = True

    if use_http:
        try:
            files = {"file": (filename, file_bytes, "audio/wav")}
            res = requests.post(f"{BACKEND_URL}/analyze", files=files, timeout=120)
            if res.status_code == 200:
                data = res.json()
        except Exception:
            data = None

    if not data or "error" in data:
        cached_clf, cached_ext, cached_wavlm = (None, None, None)
        if not is_local_backend_active(8000):
            try:
                cached_clf, cached_ext, cached_wavlm = preload_cloud_models()
            except Exception:
                pass
        data = analyze_audio_data(
            file_bytes,
            filename,
            custom_model=cached_clf,
            custom_extractor=cached_ext,
            custom_wavlm=cached_wavlm
        )

    return data


# ----------------------------------------------------------------------
# DEDICATED FORENSIC AI COPILOT DIALOG & ISOLATED SESSION CHAT HISTORY
# ----------------------------------------------------------------------
def get_file_unique_key(analysis_data: dict, filename: str) -> str:
    """Returns a unique, persistent session key for an audio file based on SHA-256 hash or filename."""
    if analysis_data and isinstance(analysis_data, dict):
        hashes = analysis_data.get("file_hashes") or {}
        sha = hashes.get("sha256")
        if sha:
            return f"sha256_{sha[:20]}"
    clean_name = os.path.basename(filename or "unknown").strip().lower()
    return f"file_{clean_name}"


def get_file_chat_history(file_key: str) -> list:
    """Retrieves or initializes the isolated chat history for an audio file."""
    if "file_chat_histories" not in st.session_state:
        st.session_state["file_chat_histories"] = {}
    if file_key not in st.session_state["file_chat_histories"]:
        st.session_state["file_chat_histories"][file_key] = []
    return st.session_state["file_chat_histories"][file_key]


def close_forensic_chat_dialog():
    """Callback triggered when the dedicated forensic chat dialog is dismissed."""
    st.session_state["active_chat_modal"] = None


@st.dialog("💬 Forensic AI Copilot — Dedicated Investigation Chatbox", width="large", on_dismiss=close_forensic_chat_dialog)
def open_forensic_chat_dialog(analysis_data: dict, audio_bytes: bytes, filename: str, gemini_key: str):
    """Renders a dedicated, high-resolution forensic investigation chat modal dialog."""
    file_key = get_file_unique_key(analysis_data, filename)
    chat_history = get_file_chat_history(file_key)

    v = (analysis_data or {}).get("verdict", "Analysis Pending")
    fake_ratio = (analysis_data or {}).get("fake_ratio", 0.0)
    hashes = (analysis_data or {}).get("file_hashes", {})
    sha = hashes.get("sha256", "")
    sha_short = (sha[:14] + "..." + sha[-6:]) if len(sha) > 20 else (sha or "N/A")
    col_v = "#3ECF8E" if "Authentic" in v or "Human" in v else "#FF5C5C" if "Synthetic" in v or "Fake" in v else "#FFB454"

    # Process auto_query if invoked via 1-click shortcut
    modal_info = st.session_state.get("active_chat_modal")
    if isinstance(modal_info, dict) and modal_info.get("auto_query"):
        auto_q = modal_info.pop("auto_query")
        chat_history.append({"role": "user", "content": auto_q})
        st.session_state["copilot_chat_history"] = chat_history
        with st.spinner("AI Copilot is processing forensic request..."):
            chat_res = chat_with_audio_copilot(
                audio_bytes=audio_bytes,
                filename=filename,
                user_query=auto_q,
                analysis=analysis_data,
                chat_history=chat_history,
                api_key=gemini_key,
            )
            if chat_res.get("success"):
                chat_history.append({"role": "assistant", "content": chat_res.get("response", "")})
                st.session_state["copilot_chat_history"] = chat_history
            else:
                st.error(chat_res.get("error", "Copilot response failed."))

    # Modal Header Card
    col_hdr_info, col_hdr_actions = st.columns([3.8, 0.9])
    with col_hdr_info:
        st.markdown(f"""
        <div style="background:#131614;border:1px solid #262B27;border-radius:8px;padding:10px 14px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
            <div>
              <div style="font-weight:700;font-size:14px;color:#ECEFEB;">📁 {filename}</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8C958E;margin-top:2px;">
                SHA-256: <code style="color:#5EEAD4;background:#1A231E;padding:2px 6px;border-radius:4px;">{sha_short}</code>
              </div>
            </div>
            <div style="text-align:right;">
              <span style="font-family:'IBM Plex Mono',monospace;font-size:11.5px;font-weight:700;color:{col_v};background:#1A1E1B;padding:4px 10px;border-radius:12px;border:1px solid rgba(255,255,255,0.06);">
                {v}
              </span>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8C958E;margin-top:3px;">
                Synthetic: <b style="color:#ECEFEB;">{fake_ratio:.1f}%</b>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_hdr_actions:
        st.write("")
        if st.button("🗑️ Clear", key=f"dlg_clear_{file_key}", use_container_width=True, help="Clear chat history for this file"):
            st.session_state["file_chat_histories"][file_key] = []
            st.session_state["copilot_chat_history"] = []
            st.rerun()

    # Pre-built quick action prompt buttons
    st.markdown('<div style="font-size:12px;font-family:\'IBM Plex Mono\',monospace;color:#8C958E;margin-bottom:6px;">QUICK FORENSIC PROMPTS:</div>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    quick_query = None
    with q1:
        if st.button("🔍 Explain Suspect Glitches", use_container_width=True, key=f"dlg_q1_{file_key}"):
            quick_query = "Please explain the highest-risk suspect window in this audio, what acoustic anomalies were triggered, and why."
    with q2:
        if st.button("⚖️ Draft Police Cyber Complaint", use_container_width=True, key=f"dlg_q2_{file_key}"):
            quick_query = "Draft an official, court-ready Cyber Crime Police Complaint (FIR Petition) under Section 173 BNSS / 154 CrPC and Section 66D IT Act for this deepfake audio recording, citing the SHA-256 hash and timestamps."
    with q3:
        if st.button("🛡️ Investigation Protocol", use_container_width=True, key=f"dlg_q3_{file_key}"):
            quick_query = "What forensic counter-verification steps and chain-of-custody protocols should be followed next for this audio?"

    # Message feed
    if not chat_history:
        st.markdown("""
        <div style="background:#151A17;border:1px dashed #2E3830;border-radius:8px;padding:22px;text-align:center;color:#8C958E;margin:12px 0;">
          <div style="font-size:24px;margin-bottom:4px;">🤖</div>
          <div style="font-size:13.5px;font-weight:600;color:#ECEFEB;">Forensic AI Copilot Ready</div>
          <div style="font-size:12px;margin-top:4px;">Ask specific questions, examine anomalous frames, or click one of the quick prompts above.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, msg in enumerate(chat_history):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🕵️"):
                    st.markdown('<b style="color:#5EEAD4;font-size:12px;font-family:\'IBM Plex Mono\',monospace;">INVESTIGATOR:</b>', unsafe_allow_html=True)
                    st.markdown(msg['content'])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown('<b style="color:#3ECF8E;font-size:12px;font-family:\'IBM Plex Mono\',monospace;">FORENSIC COPILOT:</b>', unsafe_allow_html=True)
                    st.markdown(msg['content'])

                # Check if message is a Cyber Crime Complaint or Legal Affidavit
                is_complaint = any(term in msg['content'].lower() for term in ["complaint", "affidavit", "information technology act", "cyber crime police", "station house officer", "police complaint"])
                if is_complaint:
                    try:
                        # Optional Complainant Pre-fill Form (leave blank to sign by hand with pen)
                        with st.expander("📝 Fill Complainant & Incident Details for 1-Click Print (Optional)", expanded=False):
                            st.caption("ℹ️ Pre-fill your name, contact, and police station below to embed them directly into the PDF petition. You can also leave them blank and write them by hand after printing.")
                            col_c1, col_c2 = st.columns(2)
                            with col_c1:
                                c_name = st.text_input("Complainant Full Name", key=f"c_name_dlg_{file_key}_{idx}", placeholder="e.g. Ramesh Kumar")
                                c_phone = st.text_input("Complainant Mobile", key=f"c_phone_dlg_{file_key}_{idx}", placeholder="e.g. +91 98765 43210")
                                c_addr = st.text_input("Residential Address", key=f"c_addr_dlg_{file_key}_{idx}", placeholder="e.g. Sector 18, Noida, Uttar Pradesh")
                            with col_c2:
                                c_ps = st.text_input("Police Station / Cyber Cell Name", key=f"c_ps_dlg_{file_key}_{idx}", placeholder="e.g. Cyber Crime PS, Sector 108")
                                c_susp = st.text_input("Suspect Phone / WhatsApp / Handle", key=f"c_susp_dlg_{file_key}_{idx}", placeholder="e.g. +91 91234 56789 or @scammer_handle")
                                c_date = st.text_input("Incident Date", key=f"c_date_dlg_{file_key}_{idx}", placeholder="e.g. 20-Sep-2026")

                        import importlib
                        import backend.forensic_utils as fu
                        importlib.reload(fu)
                        pdf_bytes = fu.build_police_complaint_pdf(
                            complaint_text=msg['content'],
                            filename=filename,
                            hashes=hashes,
                            verdict=v,
                            fake_ratio=fake_ratio,
                            user_tz_name=user_tz_name,
                            complainant_name=c_name,
                            complainant_phone=c_phone,
                            complainant_address=c_addr,
                            police_station=c_ps,
                            suspect_info=c_susp,
                            incident_date_str=c_date,
                        )
                        fname_base = os.path.splitext(filename)[0]
                        st.download_button(
                            label="📄 Download Official Court-Ready FIR Petition & Sec. 63 BSA Certificate (PDF)",
                            data=pdf_bytes,
                            file_name=f"Official_Cyber_Complaint_{fname_base}.pdf",
                            mime="application/pdf",
                            key=f"dl_pdf_dlg_{file_key}_{idx}",
                            type="primary",
                            use_container_width=True,
                            help="100% turnkey court-ready petition with Section 63 BSA / 65B Evidence Act certificate. Print directly and submit without needing a lawyer."
                        )
                    except Exception as e:
                        st.error(f"Could not render complaint PDF: {e}")

    # Real-time chat input
    user_input = st.chat_input("Ask the Forensic AI Copilot about this audio recording...", key=f"dlg_input_{file_key}")
    active_query = quick_query or user_input

    if active_query:
        chat_history.append({"role": "user", "content": active_query})
        st.session_state["copilot_chat_history"] = chat_history
        with st.spinner("AI Copilot is analyzing query and audio context..."):
            chat_res = chat_with_audio_copilot(
                audio_bytes=audio_bytes,
                filename=filename,
                user_query=active_query,
                analysis=analysis_data,
                chat_history=chat_history,
                api_key=gemini_key,
            )
            if chat_res.get("success"):
                chat_history.append({"role": "assistant", "content": chat_res.get("response", "")})
                st.session_state["copilot_chat_history"] = chat_history
                st.rerun()
            else:
                st.error(chat_res.get("error", "Copilot response failed."))


# Global trigger: open dedicated forensic chat modal if requested
if st.session_state.get("active_chat_modal"):
    modal_info = st.session_state["active_chat_modal"]
    open_forensic_chat_dialog(
        analysis_data=modal_info.get("data", {}),
        audio_bytes=modal_info.get("audio_bytes", None),
        filename=modal_info.get("filename", "recording.wav"),
        gemini_key=modal_info.get("gemini_key") or get_active_gemini_api_key(),
    )


# ======================================================================
# TAB 1: SINGLE AUDIO FORENSIC LOCALIZER
# ======================================================================
with tab_single:
    col_up, col_opts = st.columns([3, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload suspicious audio clip",
            type=["mp3", "wav"],
            label_visibility="collapsed",
            key="single_file_uploader"
        )

    with col_opts:
        with st.expander("⚙️ Detection Threshold", expanded=False):
            threshold_val = st.slider(
                "AI Decision Boundary",
                min_value=30,
                max_value=75,
                value=50,
                step=5,
                help="Continuous probability percentage threshold above which a window is classified as AI Fake."
            )

    if uploaded_file is not None:
        st.audio(uploaded_file)
        btn_col, _ = st.columns([1, 3])
        with btn_col:
            analyze_clicked = st.button("◈ Run Forensic Analysis", type="primary", use_container_width=True, key="btn_analyze_single")
    else:
        analyze_clicked = False

    if uploaded_file is not None and analyze_clicked:
        with st.spinner("Processing audio: Executing multi-layered forensic inspection & neural acoustic verification..."):
            uploaded_file.seek(0)
            file_bytes = uploaded_file.getvalue()
            data = execute_forensic_scan(file_bytes, uploaded_file.name)

            if not data or "error" in data:
                st.error(data.get("error", "Failed to process audio."))
                st.stop()

            st.session_state["last_result"] = data
            st.session_state["analyzed_file_name"] = uploaded_file.name
            st.session_state["analyzed_audio_bytes"] = file_bytes

            # Save to session scan history using visitor local timezone
            now_utc = datetime.now(timezone.utc)
            utc_iso = now_utc.isoformat().replace("+00:00", "Z")
            created_ts = format_local_timestamp(now_utc, user_tz_obj, user_tz_name)
            st.session_state["session_scans"].insert(0, {
                "filename": uploaded_file.name,
                "duration_sec": data.get("total_duration", 0.0),
                "fake_ratio": data.get("fake_ratio", 0.0),
                "verdict": data.get("verdict", ""),
                "created_at": created_ts,
                "created_at_utc": utc_iso,
                "result": data,
                "audio_bytes": file_bytes,
            })


    # Render Forensic Report if results available
    if "last_result" in st.session_state:
        data = st.session_state["last_result"]
        segments = data.get("segments", [])
        total_dur = data.get("total_duration", 0.0)
        fake_ratio = data.get("fake_ratio", 0.0)
        verdict = data.get("verdict", "")
        highest = data.get("highest_risk_segment", {})
        avg_conf = round(sum(s.get("confidence", 0) for s in segments) / max(len(segments), 1), 1)
        hashes = data.get("file_hashes", {})
        forensics = data.get("forensic_signals", {})
        audio_bytes = st.session_state.get("analyzed_audio_bytes", None)

        # Inspected record banner and playback if navigating from History / other tabs
        if uploaded_file is None:
            c_inf1, c_inf2, c_clr = st.columns([3, 2, 1])
            with c_inf1:
                disp_fname = st.session_state.get("analyzed_file_name", data.get("filename", "Audio File"))
                st.markdown(f"""
                <div style="background:#131614;border:1px solid #262B27;border-left:4px solid var(--cyan);padding:10px 16px;border-radius:8px;margin-bottom:12px;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#8C958E;letter-spacing:0.04em;">INSPECTING RECORD FROM SESSION</div>
                  <div style="font-size:14px;font-weight:700;color:var(--text);margin-top:2px;">📁 {disp_fname}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_inf2:
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
            with c_clr:
                st.write("")
                if st.button("✕ Clear View", key="btn_clear_inspected_view", use_container_width=True):
                    if "last_result" in st.session_state:
                        del st.session_state["last_result"]
                    if "analyzed_file_name" in st.session_state:
                        del st.session_state["analyzed_file_name"]
                    if "analyzed_audio_bytes" in st.session_state:
                        del st.session_state["analyzed_audio_bytes"]
                    st.rerun()

        # Pre-decode audio if audio bytes available for instant slicing and waveform display
        decoded_y = None
        if audio_bytes:
            try:
                decoded_y, _, _ = decode_audio_bytes(audio_bytes)
            except Exception:
                decoded_y = None

        # Verdict Pill Style & Colors
        ai_pct = round(fake_ratio, 1)
        human_pct = round(100.0 - ai_pct, 1)

        if "Authentic Human" in verdict or fake_ratio == 0:
            pill_class = "real"
            pill_text = f"✓ {verdict}"
            accent_color = "#10B981"
        elif "Fully Synthetic" in verdict or fake_ratio >= 75:
            pill_class = "fake"
            pill_text = f"⚠ {verdict}"
            accent_color = "#FF4B72"
        else:
            pill_class = "mixed"
            pill_text = f"⚡ {verdict}"
            accent_color = "#F59E0B"

        spliced_regions_list = data.get("spliced_regions", [])
        threshold_pct = threshold_val if "threshold_val" in locals() else 50

        st.markdown('<div class="section-label">◈ Forensic Biometric Triad & Executive Case Record</div>', unsafe_allow_html=True)

        # 3-Column Executive Triad (Holographic Dial | Case Record & Spliced Regions | Acoustic Radar Fingerprint)
        col_gauge, col_meta, col_radar = st.columns([1.1, 1.6, 1.3])

        with col_gauge:
            st.markdown("""
            <div style="background:var(--panel);backdrop-filter:blur(14px);border:1px solid var(--line);border-radius:14px;padding:12px 16px;text-align:center;box-shadow:0 8px 28px rgba(0,0,0,0.4);">
              <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--muted-dim);letter-spacing:0.12em;text-transform:uppercase;">SYNTHETIC RISK GAUGE</div>
            </div>
            """, unsafe_allow_html=True)
            gauge_fig = build_holographic_gauge(fake_ratio, verdict)
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"""
            <div style="text-align:center;margin-top:-22px;margin-bottom:10px;">
              <span class="pill-status {pill_class}">{pill_text}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_meta:
            peak_prob = round(highest.get("fake_probability", 0.0) * 100, 1) if highest else 0.0
            st.markdown(f"""
            <div style="background:var(--panel);backdrop-filter:blur(14px);border:1px solid var(--line);border-radius:14px;padding:16px 20px;height:100%;box-shadow:0 8px 28px rgba(0,0,0,0.4);">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--cyan);letter-spacing:0.1em;text-transform:uppercase;">CASE RECORD METRIC</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--muted);background:rgba(0,245,255,0.08);padding:3px 8px;border-radius:6px;border:1px solid rgba(0,245,255,0.2);">
                  CONFIDENCE: {avg_conf}%
                </div>
              </div>
              <div style="font-size:16px;font-weight:800;color:#FFFFFF;margin-bottom:8px;font-family:'Outfit',sans-serif;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                📁 {data.get("filename", "Audio Recording")}
              </div>
              
              <!-- Voice Composition Split Bar -->
              <div style="margin-top:12px;margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:11px;margin-bottom:5px;">
                  <span><span style="color:#10B981;font-weight:700;">● ORGANIC:</span> {human_pct}%</span>
                  <span><span style="color:{accent_color};font-weight:700;">● SYNTHETIC:</span> {ai_pct}%</span>
                </div>
                <div style="width:100%;height:8px;background:rgba(255,255,255,0.08);border-radius:6px;overflow:hidden;display:flex;">
                  <div style="width:{human_pct}%;height:100%;background:#10B981;transition:width 0.4s;"></div>
                  <div style="width:{ai_pct}%;height:100%;background:{accent_color};transition:width 0.4s;"></div>
                </div>
              </div>

              <!-- Quick Anomaly Telemetry Chips -->
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px;">
                <div style="background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:8px 10px;">
                  <div style="font-size:9.5px;color:var(--muted-dim);font-family:'JetBrains Mono';">PEAK RISK INTERVAL</div>
                  <div style="font-size:13.5px;font-weight:700;color:{'#FF4B72' if peak_prob >= 50 else '#10B981'};margin-top:2px;">
                    {peak_prob}% AI ({highest.get('start', 0)}s–{highest.get('end', 0)}s)
                  </div>
                </div>
                <div style="background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:8px 10px;">
                  <div style="font-size:9.5px;color:var(--muted-dim);font-family:'JetBrains Mono';">TEMPORAL FRAMES</div>
                  <div style="font-size:13.5px;font-weight:700;color:var(--cyan);margin-top:2px;">
                    {len(segments)} windows ({total_dur}s)
                  </div>
                </div>
              </div>

              <!-- Spliced Regions Badge -->
              <div style="margin-top:12px;">
                {'<div style="background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.35);border-radius:8px;padding:6px 12px;font-size:11.5px;color:#F59E0B;font-family:\'JetBrains Mono\';">⚠️ SPLICED AI AUDIO DETECTED: ' + str(len(spliced_regions_list)) + ' region(s)</div>' if spliced_regions_list else '<div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:8px;padding:6px 12px;font-size:11.5px;color:#10B981;font-family:\'JetBrains Mono\';">✓ UNIFIED ACOUSTIC CONTINUITY (No Splicing Detected)</div>'}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_radar:
            st.markdown("""
            <div style="background:var(--panel);backdrop-filter:blur(14px);border:1px solid var(--line);border-radius:14px;padding:12px 16px;text-align:center;box-shadow:0 8px 28px rgba(0,0,0,0.4);">
              <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--muted-dim);letter-spacing:0.12em;text-transform:uppercase;">ACOUSTIC ANOMALY RADAR</div>
            </div>
            """, unsafe_allow_html=True)
            radar_fig = build_anomaly_radar_chart(forensics, fake_ratio)
            st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

        # Detailed Spliced Insertions Callout if present
        if spliced_regions_list:
            spliced_intervals_str = ", ".join([f"<b>{r.get('start', 0)}s – {r.get('end', 0)}s</b> ({r.get('duration', 0)}s, peak {r.get('max_prob', 0)*100:.1f}% AI)" for r in spliced_regions_list])
            st.markdown(f"""
            <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.35);border-left:4px solid #F59E0B;border-radius:10px;padding:12px 18px;margin-bottom:18px;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:11.5px;color:#F59E0B;font-weight:700;">🎯 SPLICED AI SPEECH INSERTIONS LOCALIZED ({len(spliced_regions_list)} region{'s' if len(spliced_regions_list) > 1 else ''}):</div>
              <div style="font-size:12.5px;color:#F8FAFC;margin-top:4px;line-height:1.6;">
                {spliced_intervals_str}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Cryptographic Signatures & Download Action
        col_crypto, col_dl = st.columns([3, 1])
        with col_crypto:
            st.markdown(f"""
            <div class="crypto-strip">
              <div><b>SHA-256 HASH:</b> <span style="color:var(--cyan);">{hashes.get('sha256', 'N/A')}</span></div>
              <div style="color:var(--muted);"><b>File Size:</b> {hashes.get('size_kb', 0)} KB · <b>Acoustic Standard:</b> 16,000 Hz 16-bit Standardized</div>
            </div>
            """, unsafe_allow_html=True)
        with col_dl:
            report_html = generate_forensic_html_report(data, user_tz_name=user_tz_name)
            st.download_button(
                label="📥 Download Audit Report",
                data=report_html,
                file_name=f"Forensic_Report_{data.get('filename', 'scan')}.html",
                mime="text/html",
                use_container_width=True,
                help="Download an ISO-compliant, self-contained printable HTML/PDF forensic audit certificate."
            )

        # Biological & Acoustic Artifact Findings (Glassmorphic Cards)
        if forensics:
            st.markdown('<div class="section-label">◈ Biological & Acoustic Artifact Telemetry</div>', unsafe_allow_html=True)
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.markdown(f"""
                <div class="vcard">
                  <div class="k">Spectral Rolloff (85%)</div>
                  <div class="v" style="color:var(--cyan);">{forensics.get('spectral_rolloff_85_hz', 0)} <span style="font-size:13px;font-weight:500;">Hz</span></div>
                  <div class="sub">Centroid: {forensics.get('spectral_centroid_hz', 0)} Hz</div>
                </div>
                """, unsafe_allow_html=True)
            with a2:
                st.markdown(f"""
                <div class="vcard">
                  <div class="k">Zero-Crossing Rate (ZCR)</div>
                  <div class="v" style="color:#10B981;">{forensics.get('zero_crossing_rate', 0)}</div>
                  <div class="sub">Silence / Noise Boundary</div>
                </div>
                """, unsafe_allow_html=True)
            with a3:
                st.markdown(f"""
                <div class="vcard">
                  <div class="k">Dynamic Range</div>
                  <div class="v" style="color:#F8FAFC;">{forensics.get('dynamic_range_db', 0)} <span style="font-size:13px;font-weight:500;">dB</span></div>
                  <div class="sub">Crest Factor: {forensics.get('crest_factor', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            with a4:
                st.markdown(f"""
                <div class="vcard">
                  <div class="k">Pitch Inflection Score</div>
                  <div class="v" style="color:var(--amber);">{forensics.get('pitch_inflection_score', 0)}%</div>
                  <div class="sub">Mean F0: {forensics.get('pitch_mean_hz', 0)} Hz</div>
                </div>
                """, unsafe_allow_html=True)

        # Multi-Track Acoustic Waveform & Continuous Anomaly Timeline
        st.markdown('<div class="section-label">◈ Multi-Track Acoustic Waveform & Anomaly Timeline</div>', unsafe_allow_html=True)
        multi_fig = build_waveform_and_timeline_chart(
            segments=segments,
            total_dur=total_dur,
            spliced_regions=spliced_regions_list,
            decoded_y=decoded_y,
            threshold_pct=threshold_pct
        )
        st.plotly_chart(multi_fig, use_container_width=True, config={"displayModeBar": False})

        # Segment-by-Segment Forensic Explorer with Click-to-Play Audio Slicer
        st.markdown('<div class="section-label">Temporal Speech Frame Breakdown & Instant Audio Slicer</div>', unsafe_allow_html=True)

        # Pre-decode audio if audio bytes available for instant slicing
        decoded_y = None
        if audio_bytes:
            try:
                decoded_y, _, _ = decode_audio_bytes(audio_bytes)
            except Exception:
                decoded_y = None

        with st.expander(f"Inspect all {len(segments)} Temporal Speech Frames", expanded=False):
            seg_cols = st.columns(3)
            for idx, s in enumerate(segments):
                with seg_cols[idx % 3]:
                    is_fake = s["fake_probability"] * 100 >= threshold_pct
                    color = "#FF5C5C" if is_fake else "#3ECF8E"
                    st.markdown(f"""
                    <div style="background:#1A1E1B;border:1px solid #262B27;border-left:3px solid {color};padding:10px 12px;border-radius:8px;margin-bottom:8px;">
                      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#8C958E;">
                        FRAME #{s.get('segment', idx) + 1} · {s.get('start', 0)}s–{s.get('end', 0)}s
                      </div>

                      <div style="font-weight:700;font-size:14px;color:{color};margin-top:2px;">
                        {'AI Fake' if is_fake else 'Human'} ({s['fake_probability']*100:.1f}%)
                      </div>
                      <div style="font-size:11px;color:#5E6660;">Confidence: {s['confidence']}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Click-to-Play Audio Slice
                    if decoded_y is not None:
                        try:
                            seg_audio_bytes = slice_segment_audio_bytes(decoded_y, s['start'], s['end'], sr=16000)
                            st.audio(seg_audio_bytes, format="audio/wav")
                        except Exception:
                            pass

        # 3D Voice Signature Spectrogram
        st.markdown('<div class="section-label">◈ 3D Voice Signature Spectrogram (Acoustic Topography)</div>', unsafe_allow_html=True)
        spec_data = data.get("spectrogram", {})
        if "z" in spec_data and spec_data["z"]:
            z = np.array(spec_data["z"])
            x = np.array(spec_data.get("time", list(range(z.shape[1]))))
            y_axis = np.array(spec_data.get("mel", list(range(z.shape[0]))))

            fig3d = go.Figure(data=[go.Surface(
                z=z, x=x, y=y_axis,
                colorscale=[
                    [0, "#07090E"],
                    [0.25, "#0D1E36"],
                    [0.55, "#0E486D"],
                    [0.8, "#00C4D4"],
                    [1.0, "#00F5FF"]
                ],
                showscale=False
            )])
            fig3d.update_layout(
                scene=dict(
                    xaxis=dict(title="", showbackground=False, color="#64748B"),
                    yaxis=dict(title="", showbackground=False, color="#64748B"),
                    zaxis=dict(title="", showbackground=False, color="#64748B"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=380,
            )
            st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": False})

        # ----------------------------------------------------------------------
        # ◈ GEMINI AI FORENSIC INTELLIGENCE & COPILOT SUITE
        # ----------------------------------------------------------------------
        st.markdown('<div class="section-label" style="margin-top:35px;">◈ Gemini AI Forensic Intelligence & Copilot</div>', unsafe_allow_html=True)

        gemini_key = get_active_gemini_api_key()
        has_key = bool(gemini_key)

        status_tag = (
            '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#3ECF8E;background:rgba(62,207,142,0.1);padding:4px 10px;border-radius:16px;border:1px solid rgba(62,207,142,0.3);">● AI ENGINE READY</span>'
            if has_key
            else '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#FFB454;background:rgba(255,180,84,0.1);padding:4px 10px;border-radius:16px;border:1px solid rgba(255,180,84,0.3);">○ API KEY REQUIRED</span>'
        )

        st.markdown(f"""
        <div style="background:#111513;border:1px solid #232B25;border-radius:12px;padding:16px 20px;margin-bottom:20px;">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
            <div>
              <div style="font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:15px;color:#ECEFEB;display:flex;align-items:center;gap:8px;">
                <span style="color:#5EEAD4;">🤖</span> Multimodal Forensic Reasoning & Threat Intel
              </div>
              <div style="font-size:12px;color:#8C958E;margin-top:3px;">
                Correlates WavLM deep acoustic anomalies with speech semantics using Google Gemini 3.6 Flash.
              </div>
            </div>
            <div>
              {status_tag}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not has_key:
            st.info("💡 **Enable Next-Gen AI Insights**: Enter your free Gemini API Key below to unlock **AI Forensic Explanations**, **Scam Threat Intent Detection**, and the **Interactive Audio Copilot**.")
            c_key_in, c_key_btn = st.columns([3, 1])
            with c_key_in:
                direct_key_input = st.text_input(
                    "Paste Gemini API Key",
                    type="password",
                    placeholder="Paste your Gemini API key (AIzaSy...)",
                    key="direct_gemini_key_input",
                    label_visibility="collapsed",
                )
            with c_key_btn:
                if st.button("Connect AI Engine", key="btn_connect_gemini", type="primary", use_container_width=True):
                    if direct_key_input.strip():
                        st.session_state["custom_gemini_api_key"] = direct_key_input.strip()
                        st.rerun()
                    else:
                        st.warning("Please enter an API key.")
            st.markdown(
                """<div style="font-size:12px;color:#8C958E;margin-top:4px;margin-bottom:15px;">
                Don't have a key? Get one 100% free in 30 seconds at <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#3ECF8E;text-decoration:none;font-weight:600;">Google AI Studio ↗</a> (No credit card required).
                </div>""",
                unsafe_allow_html=True
            )
        else:
            sub_ai_tab1, sub_ai_tab2, sub_ai_tab3 = st.tabs([
                "📝 Forensic Reasoning (XAI)",
                "🚨 Scam & Threat Intent",
                "💬 Interactive Copilot Chat"
            ])

            with sub_ai_tab1:
                st.markdown('<div style="font-size:13px;color:#8C958E;margin-bottom:12px;">Generates an authoritative, plain-language forensic briefing explaining the spectral rolloff, pitch micro-inflections, and vocoder artifacts.</div>', unsafe_allow_html=True)
                col_lang, col_btn = st.columns([1, 2])
                with col_lang:
                    lang_choice = st.selectbox("Briefing Language", ["English", "Hindi / Hinglish"], key="ai_lang_select")
                with col_btn:
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    gen_xai_clicked = st.button("◈ Generate Forensic Reasoning Report", key="btn_gen_xai", type="primary")

                if gen_xai_clicked:
                    with st.spinner("Gemini is synthesizing forensic acoustic vectors & vocoder signatures..."):
                        fname = st.session_state.get("analyzed_file_name", data.get("filename", "recording.wav"))
                        xai_res = generate_forensic_explanation(
                            audio_bytes=audio_bytes,
                            filename=fname,
                            analysis=data,
                            language=lang_choice,
                            api_key=gemini_key,
                        )
                        if xai_res.get("success"):
                            st.session_state["xai_report"] = xai_res.get("explanation", "")
                        else:
                            st.error(xai_res.get("error", "Failed to generate report."))

                if "xai_report" in st.session_state and st.session_state["xai_report"]:
                    st.markdown(f"""
                    <div style="background:#0D110E;border:1px solid rgba(62,207,142,0.3);border-radius:10px;padding:20px;margin-top:12px;color:#ECEFEB;line-height:1.6;">
                    {st.session_state['xai_report']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.download_button(
                        label="⬇️ Download AI Forensic Brief (.md)",
                        data=st.session_state["xai_report"],
                        file_name=f"Forensic_AI_Brief_{st.session_state.get('analyzed_file_name', 'audio')}.md",
                        mime="text/markdown",
                        key="btn_dl_xai"
                    )

            with sub_ai_tab2:
                st.markdown('<div style="font-size:13px;color:#8C958E;margin-bottom:12px;">Transcribes spoken speech and performs semantic fraud detection (financial urgency, impersonation extortion, CEO fraud, family distress scams).</div>', unsafe_allow_html=True)
                if st.button("🚨 Run Semantic Scam & Threat Assessment", key="btn_run_threat", type="primary"):
                    with st.spinner("Transcribing speech and screening for social engineering / scam vectors..."):
                        fname = st.session_state.get("analyzed_file_name", data.get("filename", "recording.wav"))
                        threat_res = assess_scam_threat(
                            audio_bytes=audio_bytes,
                            filename=fname,
                            analysis=data,
                            api_key=gemini_key,
                        )
                        if threat_res.get("success"):
                            st.session_state["threat_report"] = threat_res.get("assessment", "")
                        else:
                            st.error(threat_res.get("error", "Failed to assess threat."))

                if "threat_report" in st.session_state and st.session_state["threat_report"]:
                    st.markdown(f"""
                    <div style="background:#140F0E;border:1px solid rgba(255,87,87,0.3);border-radius:10px;padding:20px;margin-top:12px;color:#ECEFEB;line-height:1.6;">
                    {st.session_state['threat_report']}
                    </div>
                    """, unsafe_allow_html=True)

            with sub_ai_tab3:
                fname = st.session_state.get("analyzed_file_name", data.get("filename", "recording.wav"))
                curr_file_key = get_file_unique_key(data, fname)
                curr_chat = get_file_chat_history(curr_file_key)
                msg_count = len(curr_chat)
                user_msg_count = len([m for m in curr_chat if m.get("role") == "user"])

                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #131A16 0%, #0E1210 100%);border:1px solid #233027;border-radius:10px;padding:20px 22px;margin-bottom:16px;">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
                    <div>
                      <div style="display:flex;align-items:center;gap:8px;">
                        <span style="font-size:20px;">💬</span>
                        <span style="font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:700;color:#ECEFEB;">Forensic AI Copilot — Dedicated Investigation Chatbox</span>
                      </div>
                      <div style="color:#8C958E;font-size:13px;margin-top:6px;max-width:680px;line-height:1.45;">
                        Interact directly with your audio sample in an isolated, high-resolution modal chatbox. Ask deep technical questions, isolate frame glitches, cross-examine synthetic artifacts, or generate official court FIR affidavits with cryptographic chain of custody.
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <span style="font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:#3ECF8E;background:rgba(62,207,142,0.12);padding:4px 10px;border-radius:12px;border:1px solid rgba(62,207,142,0.25);">
                        ● {user_msg_count} Questions Logged
                      </span>
                      <div style="font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#8C958E;margin-top:4px;">
                        File: <code style="color:#5EEAD4;">{fname}</code>
                      </div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Primary Action to open modal
                col_main_btn, col_help = st.columns([2.8, 1.2])
                with col_main_btn:
                    if st.button(
                        "💬 Open Dedicated AI Investigation Chatbox",
                        key=f"btn_launch_dialog_{curr_file_key}",
                        type="primary",
                        use_container_width=True,
                        help="Open high-resolution modal dialog for multi-turn forensic investigation."
                    ):
                        st.session_state["active_chat_modal"] = {
                            "data": data,
                            "audio_bytes": audio_bytes,
                            "filename": fname,
                            "gemini_key": gemini_key,
                        }
                        st.rerun()

                with col_help:
                    if st.button("🗑️ Reset File Chat", key=f"btn_reset_chat_{curr_file_key}", use_container_width=True, disabled=(msg_count == 0)):
                        st.session_state["file_chat_histories"][curr_file_key] = []
                        st.session_state["copilot_chat_history"] = []
                        st.rerun()

                # 1-Click Quick Action shortcuts
                st.markdown('<div style="font-size:12px;font-family:\'IBM Plex Mono\',monospace;color:#8C958E;margin:14px 0 6px 0;">1-CLICK DIRECT FORENSIC ACTIONS:</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("🔍 Explain Suspect Glitches", use_container_width=True, key="quick_open_q1"):
                        st.session_state["active_chat_modal"] = {
                            "data": data,
                            "audio_bytes": audio_bytes,
                            "filename": fname,
                            "gemini_key": gemini_key,
                            "auto_query": "Please explain the highest-risk suspect window in this audio, what acoustic anomalies were triggered, and why."
                        }
                        st.rerun()
                with c2:
                    if st.button("⚖️ Draft Police Cyber Complaint", use_container_width=True, key="quick_open_q2"):
                        st.session_state["active_chat_modal"] = {
                            "data": data,
                            "audio_bytes": audio_bytes,
                            "filename": fname,
                            "gemini_key": gemini_key,
                            "auto_query": "Draft an official, court-ready Cyber Crime Police Complaint (FIR Petition) under Section 173 BNSS / 154 CrPC and Section 66D IT Act for this deepfake audio recording, citing the SHA-256 hash and timestamps."
                        }
                        st.rerun()
                with c3:
                    if st.button("🛡️ Investigation Protocol", use_container_width=True, key="quick_open_q3"):
                        st.session_state["active_chat_modal"] = {
                            "data": data,
                            "audio_bytes": audio_bytes,
                            "filename": fname,
                            "gemini_key": gemini_key,
                            "auto_query": "What forensic counter-verification steps and chain-of-custody protocols should be followed next for this audio?"
                        }
                        st.rerun()

                # Expandable preview of recent messages if any exist
                if msg_count > 0:
                    with st.expander(f"📜 View Conversation Preview ({user_msg_count} questions logged)", expanded=False):
                        st.caption("Read-only snapshot of conversation history for this file. Open the dedicated chatbox above to continue conversation.")
                        for idx, msg in enumerate(curr_chat):
                            if msg["role"] == "user":
                                with st.chat_message("user", avatar="🕵️"):
                                    st.markdown('<b style="color:#5EEAD4;font-size:11.5px;font-family:\'IBM Plex Mono\',monospace;">INVESTIGATOR:</b>', unsafe_allow_html=True)
                                    st.markdown(msg['content'])
                            else:
                                with st.chat_message("assistant", avatar="🤖"):
                                    st.markdown('<b style="color:#3ECF8E;font-size:11.5px;font-family:\'IBM Plex Mono\',monospace;">FORENSIC COPILOT:</b>', unsafe_allow_html=True)
                                    st.markdown(msg['content'])

                                is_complaint = any(term in msg['content'].lower() for term in ["complaint", "affidavit", "information technology act", "cyber crime police", "station house officer", "police complaint"])
                                if is_complaint:
                                    try:
                                        # Optional Complainant Pre-fill Form (leave blank to sign by hand with pen)
                                        with st.expander("📝 Fill Complainant & Incident Details for 1-Click Print (Optional)", expanded=False):
                                            st.caption("ℹ️ Pre-fill your name, contact, and police station below to embed them directly into the PDF petition. You can also leave them blank and write them by hand after printing.")
                                            col_c1, col_c2 = st.columns(2)
                                            with col_c1:
                                                c_name = st.text_input("Complainant Full Name", key=f"c_name_prev_{curr_file_key}_{idx}", placeholder="e.g. Ramesh Kumar")
                                                c_phone = st.text_input("Complainant Mobile", key=f"c_phone_prev_{curr_file_key}_{idx}", placeholder="e.g. +91 98765 43210")
                                                c_addr = st.text_input("Residential Address", key=f"c_addr_prev_{curr_file_key}_{idx}", placeholder="e.g. Sector 18, Noida, Uttar Pradesh")
                                            with col_c2:
                                                c_ps = st.text_input("Police Station / Cyber Cell Name", key=f"c_ps_prev_{curr_file_key}_{idx}", placeholder="e.g. Cyber Crime PS, Sector 108")
                                                c_susp = st.text_input("Suspect Phone / WhatsApp / Handle", key=f"c_susp_prev_{curr_file_key}_{idx}", placeholder="e.g. +91 91234 56789 or @scammer_handle")
                                                c_date = st.text_input("Incident Date", key=f"c_date_prev_{curr_file_key}_{idx}", placeholder="e.g. 20-Sep-2026")

                                        import importlib
                                        import backend.forensic_utils as fu
                                        importlib.reload(fu)
                                        pdf_bytes = fu.build_police_complaint_pdf(
                                            complaint_text=msg['content'],
                                            filename=fname,
                                            hashes=data.get("file_hashes", {}),
                                            verdict=data.get("verdict", "Unknown"),
                                            fake_ratio=data.get("fake_ratio", 0.0),
                                            user_tz_name=user_tz_name,
                                            complainant_name=c_name,
                                            complainant_phone=c_phone,
                                            complainant_address=c_addr,
                                            police_station=c_ps,
                                            suspect_info=c_susp,
                                            incident_date_str=c_date,
                                        )
                                        fname_base = os.path.splitext(fname)[0]
                                        st.download_button(
                                            label="📄 Download Official Court-Ready FIR Petition & Sec. 63 BSA Certificate (PDF)",
                                            data=pdf_bytes,
                                            file_name=f"Official_Cyber_Complaint_{fname_base}.pdf",
                                            mime="application/pdf",
                                            key=f"dl_pdf_preview_{curr_file_key}_{idx}",
                                            type="primary",
                                            use_container_width=True,
                                            help="100% turnkey court-ready petition with Section 63 BSA / 65B Evidence Act certificate. Print directly and submit without needing a lawyer."
                                        )
                                    except Exception as e:
                                        st.error(f"Could not render PDF: {e}")


# ======================================================================
# TAB 2: LIVE MICROPHONE VOICE TEST
# ======================================================================
with tab_mic:
    st.markdown('<div class="section-label">Live Voice Authenticity Test</div>', unsafe_allow_html=True)
    st.markdown("""
    Test your real voice or a live played speaker sample directly using your microphone.
    The system captures your recording, isolates active vocal frequency zones,
    and screens for synthetic speech artifacts in real time.
    """)

    mic_audio = st.audio_input("Record speech via microphone", key="mic_recorder")

    if mic_audio is not None:
        st.audio(mic_audio)
        if st.button("◈ Analyze Live Recording", type="primary", key="btn_analyze_mic"):
            with st.spinner("Analyzing live microphone audio across complete duration..."):
                mic_bytes = mic_audio.getvalue()
                data = execute_forensic_scan(mic_bytes, "live_microphone_recording.wav")

                if not data or "error" in data:
                    st.error(data.get("error", "Error processing microphone recording."))
                else:
                    st.session_state["last_result"] = data
                    st.session_state["analyzed_file_name"] = "live_microphone_recording.wav"
                    st.session_state["analyzed_audio_bytes"] = mic_bytes
                    st.session_state["mic_analyzed_data"] = data
                    st.session_state["mic_analyzed_bytes"] = mic_bytes

                    now_utc = datetime.now(timezone.utc)
                    utc_iso = now_utc.isoformat().replace("+00:00", "Z")
                    created_ts = format_local_timestamp(now_utc, user_tz_obj, user_tz_name)
                    st.session_state["session_scans"].insert(0, {
                        "filename": "live_microphone_recording.wav",
                        "duration_sec": data.get("total_duration", 0.0),
                        "fake_ratio": data.get("fake_ratio", 0.0),
                        "verdict": data.get("verdict", ""),
                        "created_at": created_ts,
                        "created_at_utc": utc_iso,
                        "result": data,
                        "audio_bytes": mic_bytes,
                    })

        # Display results if available for the current microphone recording
        if "mic_analyzed_data" in st.session_state and mic_audio is not None:
            data = st.session_state["mic_analyzed_data"]
            mic_bytes = st.session_state.get("mic_analyzed_bytes", mic_audio.getvalue())
            segments = data.get("segments", [])
            total_dur = data.get("total_duration", 0.0)
            fake_ratio = data.get("fake_ratio", 0.0)
            verdict = data.get("verdict", "")
            col_v = "#3ECF8E" if ("Authentic Human" in verdict or fake_ratio == 0) else "#FF5C5C" if ("Fully Synthetic" in verdict or fake_ratio >= 75) else "#FFB454"

            mic_ai_pct = round(fake_ratio, 1)
            mic_human_pct = round(100.0 - mic_ai_pct, 1)

            st.markdown(f"""
            <div style="background:#131614;border:1px solid #262B27;border-left:4px solid {col_v};padding:18px;border-radius:10px;margin-top:14px;margin-bottom:14px;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8C958E;letter-spacing:0.04em;">LIVE MIC VERDICT · FULL TIMELINE AUDIT</div>
              <div style="font-size:22px;font-weight:700;color:{col_v};margin:4px 0;">{verdict}</div>
              <div style="font-size:13px;color:#ECEFEB;margin-top:6px;">
                Overall AI Content: <b style="color:{'#3ECF8E' if mic_ai_pct == 0 else '#FF5C5C'};font-size:15px;">{mic_ai_pct}% AI</b> {f'🛡️ (100% Pure Organic Voice)' if mic_ai_pct == 0 else f'({mic_human_pct}% Human)'} · Total Recorded: <b>{total_dur}s</b> · Analyzed Frames: <b>{len(segments)} Temporal Windows</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Frame-by-frame breakdown directly in Live Mic
            if segments:
                st.markdown('<div class="section-label">Continuous Temporal Windows (0.0s to End)</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(segments), 3))
                for idx, s in enumerate(segments):
                    with cols[idx % len(cols)]:
                        is_seg_fake = s["fake_probability"] >= 0.5
                        seg_col = "#FF5C5C" if is_seg_fake else "#3ECF8E"
                        st.markdown(f"""
                        <div style="background:#1A1E1B;border:1px solid #262B27;border-left:3px solid {seg_col};padding:8px 12px;border-radius:8px;margin-bottom:8px;">
                          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#8C958E;">
                            WINDOW #{idx+1} · {s.get('start', 0.0)}s – {s.get('end', 0.0)}s
                          </div>
                          <div style="font-weight:700;font-size:13px;color:{seg_col};margin-top:2px;">
                            {'AI Fake' if is_seg_fake else 'Human'} ({s.get('fake_probability', 0)*100:.1f}%)
                          </div>
                          <div style="font-size:10.5px;color:#5E6660;">Confidence: {s.get('confidence', 0)}%</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.button(
                "◈ Inspect Deep 3D Spectrogram & Slices in Single Localizer",
                key="btn_mic_goto_single",
                on_click=reinspect_scan_callback,
                args=(data, "live_microphone_recording.wav", mic_bytes),
                type="primary",
                use_container_width=True
            )


# ======================================================================
# TAB 3: SPEAKER CLONE / VOICEPRINT MATCHER
# ======================================================================
with tab_speaker:
    st.markdown('<div class="section-label">Speaker Verification & Clone Impersonation Audit</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a **Reference Audio (Known Authentic Voice)** and a **Suspect Audio (Alleged Voice Clone or Recording)**.
    The system extracts deep neural voiceprints to calculate **Speaker Voiceprint Acoustic Similarity**
    and independently determines if the suspect clip contains synthetic voice cloning artifacts.
    """)

    col_ref, col_sus = st.columns(2)
    with col_ref:
        st.markdown("##### 1. Reference Audio (Known Genuine Voice)")
        ref_file = st.file_uploader("Upload reference audio", type=["wav", "mp3"], key="ref_file_uploader")
        if ref_file:
            st.audio(ref_file)

    with col_sus:
        st.markdown("##### 2. Suspect Audio (Alleged Clone)")
        sus_file = st.file_uploader("Upload suspect audio", type=["wav", "mp3"], key="sus_file_uploader")
        if sus_file:
            st.audio(sus_file)

    if ref_file and sus_file:
        if st.button("◈ Run Voiceprint Comparison & Clone Audit", type="primary", key="btn_compare_speakers"):
            with st.spinner("Extracting neural voiceprints and analyzing acoustic concordance..."):

                ref_bytes = ref_file.getvalue()
                sus_bytes = sus_file.getvalue()

                cached_clf, cached_ext, cached_wavlm = (None, None, None)
                if not is_local_backend_active(8000):
                    try:
                        cached_clf, cached_ext, cached_wavlm = preload_cloud_models()
                    except Exception:
                        pass

                emb_ref, err_ref = extract_speaker_embedding_vector(
                    ref_bytes, ref_file.name, custom_extractor=cached_ext, custom_wavlm=cached_wavlm
                )
                emb_sus, err_sus = extract_speaker_embedding_vector(
                    sus_bytes, sus_file.name, custom_extractor=cached_ext, custom_wavlm=cached_wavlm
                )

                if err_ref:
                    st.error(f"Reference Audio Error: {err_ref}")
                elif err_sus:
                    st.error(f"Suspect Audio Error: {err_sus}")
                else:
                    sim_data = compute_speaker_similarity(emb_ref, emb_sus)

                    # Also analyze suspect audio for deepfake probability
                    data_sus = execute_forensic_scan(sus_bytes, sus_file.name)
                    sus_fake_ratio = data_sus.get("fake_ratio", 0.0) if data_sus else 0.0
                    is_sus_ai = sus_fake_ratio >= 50.0

                    sim_pct = sim_data.get("similarity_percentage", 0.0)
                    cos_sim = sim_data.get("cosine_similarity", 0.0)

                    # Determine Dual Verdict
                    if sim_pct >= 82.0 and is_sus_ai:
                        alert_title = "🚨 HIGH-RISK VOICE CLONE DETECTED"
                        alert_desc = f"The suspect audio matches the vocal identity of the reference speaker ({sim_pct}% voiceprint match), but contains {sus_fake_ratio}% synthetic speech segments. High probability of AI voice cloning / impersonation attack."
                        alert_color = "#FF5C5C"
                    elif sim_pct >= 82.0 and not is_sus_ai:
                        alert_title = "✓ AUTHENTIC SPEAKER MATCH"
                        alert_desc = f"The suspect audio matches the reference speaker ({sim_pct}% voiceprint match) and is classified as authentic human voice ({sus_fake_ratio}% AI ratio)."
                        alert_color = "#3ECF8E"
                    elif sim_pct < 65.0:
                        alert_title = "❌ DISTINCT SPEAKERS"
                        alert_desc = f"The vocal tracts and acoustic resonances do not match ({sim_pct}% similarity). The recordings represent two completely different speakers."
                        alert_color = "#8C958E"
                    else:
                        alert_title = "⚠️ AMBIGUOUS / PARTIAL SIMILARITY"
                        alert_desc = f"Moderate acoustic overlap ({sim_pct}% match). Inconclusive identity match; check recording quality and background noise."
                        alert_color = "#FFB454"

                    st.markdown(f"""
                    <div style="background:#131614;border:1px solid #262B27;border-left:5px solid {alert_color};padding:22px;border-radius:12px;margin:20px 0;">
                      <div style="font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:{alert_color};font-weight:700;">{alert_title}</div>
                      <div style="font-size:15px;color:#ECEFEB;margin-top:8px;line-height:1.6;">{alert_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Metric Breakdown Cards
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""
                        <div class="vcard">
                          <div class="k">Voiceprint Similarity</div>
                          <div class="v" style="color:var(--cyan);">{sim_pct}%</div>
                          <div class="sub">Cosine Score: {cos_sim}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="vcard">
                          <div class="k">Suspect Synthetic Ratio</div>
                          <div class="v" style="color:{'#FF5C5C' if sus_fake_ratio >= 50 else '#3ECF8E'};">{sus_fake_ratio}%</div>
                          <div class="sub">Verdict: {data_sus.get('verdict', 'N/A')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="vcard">
                          <div class="k">Embedding Distance</div>
                          <div class="v">{sim_data.get('euclidean_distance', 0)}</div>
                          <div class="sub">Euclidean vector divergence</div>
                        </div>
                        """, unsafe_allow_html=True)

                    if data_sus:
                        now_utc = datetime.now(timezone.utc)
                        utc_iso = now_utc.isoformat().replace("+00:00", "Z")
                        created_ts = format_local_timestamp(now_utc, user_tz_obj, user_tz_name)
                        st.session_state["session_scans"].insert(0, {
                            "filename": f"[Suspect] {sus_file.name}",
                            "duration_sec": data_sus.get("total_duration", 0.0),
                            "fake_ratio": sus_fake_ratio,
                            "verdict": data_sus.get("verdict", ""),
                            "created_at": created_ts,
                            "created_at_utc": utc_iso,
                            "result": data_sus,
                            "audio_bytes": sus_bytes,
                        })

                        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
                        c_spk_info, c_spk_btn = st.columns([3, 1])
                        with c_spk_info:
                            st.markdown(f"""
                            <div style="background:#131614;border:1px solid #262B27;border-left:4px solid var(--cyan);padding:12px 16px;border-radius:8px;">
                              <div style="font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#8C958E;letter-spacing:0.04em;">DEEP FORENSIC TIMELINE</div>
                              <div style="font-size:13.5px;color:#ECEFEB;margin-top:2px;">
                                Inspect <b>{sus_file.name}</b> in Single Localizer to view temporal speech frames, probability curves, and 3D spectrogram.
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with c_spk_btn:
                            st.write("")
                            st.button(
                                "Inspect Suspect ◈",
                                key="btn_inspect_suspect_clone",
                                type="primary",
                                use_container_width=True,
                                on_click=reinspect_scan_callback,
                                args=(data_sus, sus_file.name, sus_bytes)
                            )


# ======================================================================
# TAB 4: BATCH FORENSIC SCANNER
# ======================================================================
with tab_batch:
    st.markdown('<div class="section-label">Batch Forensic Scanner</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload multiple audio files at once to perform automated bulk deepfake screening.
    Review comparative metrics, download aggregated CSV audit reports, and inspect individual files.
    """)

    batch_files = st.file_uploader(
        "Upload multiple audio files",
        type=["wav", "mp3"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if batch_files:
        st.write(f"Selected **{len(batch_files)}** audio files.")
        if st.button(f"◈ Run Bulk Forensic Scan ({len(batch_files)} files)", type="primary", key="btn_run_batch"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            batch_list = []

            for i, f in enumerate(batch_files):
                status_text.text(f"Scanning [{i+1}/{len(batch_files)}]: {f.name}...")
                f.seek(0)
                content = f.getvalue()
                res = execute_forensic_scan(content, f.name)

                if res and "error" not in res:
                    f_dur = res.get("total_duration", 0.0)
                    f_ratio = res.get("fake_ratio", 0.0)
                    f_verdict = res.get("verdict", "N/A")
                    f_hash = res.get("file_hashes", {}).get("sha256", "")[:16] + "..."
                    peak = res.get("highest_risk_segment", {}).get("fake_probability", 0.0) * 100

                    now_utc = datetime.now(timezone.utc)
                    utc_iso = now_utc.isoformat().replace("+00:00", "Z")
                    b_ts = format_local_timestamp(now_utc, user_tz_obj, user_tz_name)

                    batch_list.append({
                        "Filename": f.name,
                        "Duration (s)": f_dur,
                        "Synthetic Ratio (%)": f_ratio,
                        "Peak Risk (%)": round(peak, 1),
                        "Verdict": f_verdict,
                        "Scanned At": b_ts,
                        "SHA-256 Preview": f_hash,
                        "_raw_result": res,
                        "_raw_bytes": content,
                        "_created_at_utc": utc_iso,
                    })

                    st.session_state["session_scans"].insert(0, {
                        "filename": f.name,
                        "duration_sec": f_dur,
                        "fake_ratio": f_ratio,
                        "verdict": f_verdict,
                        "created_at": b_ts,
                        "created_at_utc": utc_iso,
                        "result": res,
                        "audio_bytes": content,
                    })

                progress_bar.progress((i + 1) / len(batch_files))

            status_text.text(f"✓ Bulk scanning completed for {len(batch_list)} files.")
            st.session_state["batch_results"] = batch_list

    if st.session_state.get("batch_results"):
        b_results = st.session_state["batch_results"]
        df_display = pd.DataFrame([
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in b_results
        ])

        st.dataframe(df_display, use_container_width=True)

        # CSV Download
        csv_buffer = io.StringIO()
        df_display.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Batch Summary (CSV)",
            data=csv_buffer.getvalue(),
            file_name="AudioArtifact_Batch_Forensic_Summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        # One-click inspect any batch item in Single Localizer
        st.markdown("##### ◈ Inspect File in Single Localizer")
        c_bsel, c_bact = st.columns([3, 1])
        with c_bsel:
            batch_filenames = [r["Filename"] for r in b_results]
            selected_b_file = st.selectbox(
                "Select a scanned file to inspect its full timeline:",
                options=batch_filenames,
                key="sel_batch_inspect"
            )
        with c_bact:
            st.write("")
            sel_item = next((r for r in b_results if r["Filename"] == selected_b_file), None)
            if sel_item:
                st.button(
                    "Inspect ◈",
                    key="btn_batch_inspect_redirect",
                    use_container_width=True,
                    on_click=reinspect_scan_callback,
                    args=(sel_item["_raw_result"], sel_item["Filename"], sel_item.get("_raw_bytes"))
                )


# ======================================================================
# TAB 5: SESSION HISTORY & AUDIT LOG
# ======================================================================
with tab_history:
    st.markdown('<div class="section-label">Session Forensic History & Audit Log</div>', unsafe_allow_html=True)

    session_scans = st.session_state.get("session_scans", [])

    col_h_ctrl1, col_h_ctrl2, col_h_tz, col_h_clear = st.columns([2, 2, 2, 1])
    with col_h_ctrl1:
        search_query = st.text_input("🔍 Search by filename", "", placeholder="Enter file name...", key="hist_search")
    with col_h_ctrl2:
        verdict_filter = st.selectbox("Filter by Verdict", ["All", "AI Fake / Synthetic", "Authentic Human", "Spliced Audio"], key="hist_filter")
    with col_h_tz:
        tz_labels = [label for label, _ in COMMON_TIMEZONES]
        tz_values = [val for _, val in COMMON_TIMEZONES]
        current_override = st.session_state.get("user_tz_override", "AUTO")
        default_idx = tz_values.index(current_override) if current_override in tz_values else 0
        selected_tz_label = st.selectbox(
            "🌐 Display Timezone",
            options=tz_labels,
            index=default_idx,
            key="sel_hist_timezone"
        )
        new_override_val = tz_values[tz_labels.index(selected_tz_label)]
        if new_override_val != st.session_state.get("user_tz_override"):
            st.session_state["user_tz_override"] = new_override_val
            st.rerun()
    with col_h_clear:
        st.write("")
        st.write("")
        if session_scans and st.button("🗑️ Clear History", use_container_width=True, key="btn_clear_hist"):
            st.session_state["session_scans"] = []
            if "last_result" in st.session_state:
                del st.session_state["last_result"]
            st.rerun()

    # Filter records
    filtered_scans = []
    for s in session_scans:
        v = s.get("verdict", "")
        f_name = s.get("filename", "").lower()
        if search_query.lower() and search_query.lower() not in f_name:
            continue
        if verdict_filter == "AI Fake / Synthetic" and not ("Synthetic" in v or "Fake" in v):
            continue
        if verdict_filter == "Authentic Human" and not ("Authentic" in v or "Human" in v):
            continue
        if verdict_filter == "Spliced Audio" and "Spliced" not in v:
            continue
        filtered_scans.append(s)

    if filtered_scans:
        st.write(f"Showing **{len(filtered_scans)}** scan record{'s' if len(filtered_scans) > 1 else ''} · Timezone: **{user_tz_label}**:")
        for idx, h in enumerate(filtered_scans):
            v = h.get("verdict", "")
            col_v = "#3ECF8E" if "Authentic" in v or "Human" in v else "#FF5C5C" if "Synthetic" in v or "Fake" in v else "#FFB454"
            utc_iso = h.get("created_at_utc", "")
            disp_time = format_local_timestamp(utc_iso, user_tz_obj, user_tz_name) if utc_iso else h.get("created_at", "")
            c_info, c_inspect, c_chat = st.columns([3.3, 0.9, 1.2])
            with c_info:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;background:#131614;border:1px solid #262B27;border-radius:8px;padding:12px 16px;margin-bottom:6px;">
                  <div>
                    <div style="font-weight:600;font-size:13.5px;">{h.get('filename')}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8C958E;margin-top:3px;">
                      Duration: {h.get('duration_sec', 0):.1f}s · Scanned at <span class="client-local-time" data-utc="{utc_iso}">{disp_time}</span>
                    </div>

                  </div>
                  <div style="text-align:right;">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:11.5px;font-weight:600;color:{col_v};background:#1A1E1B;padding:4px 10px;border-radius:12px;">
                      {v or 'Scanned'}
                    </span>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8C958E;margin-top:3px;">
                      Synthetic: {h.get('fake_ratio', 0):.1f}%
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with c_inspect:
                st.write("")
                st.button(
                    "Inspect ◈",
                    key=f"reinspect_tab_{idx}",
                    use_container_width=True,
                    on_click=reinspect_scan_callback,
                    args=(h.get("result", {}), h.get("filename", ""), h.get("audio_bytes", None))
                )
            with c_chat:
                st.write("")
                h_data = h.get("result", {})
                h_name = h.get("filename", "")
                h_key = get_file_unique_key(h_data, h_name)
                h_history = get_file_chat_history(h_key)
                h_count = len([m for m in h_history if m.get("role") == "user"])
                chat_label = f"💬 Chat ({h_count})" if h_count > 0 else "💬 Chat History"
                if st.button(chat_label, key=f"open_chat_tab5_{idx}", use_container_width=True, help=f"Open dedicated AI Copilot chat history for {h_name}"):
                    st.session_state["active_chat_modal"] = {
                        "data": h_data,
                        "audio_bytes": h.get("audio_bytes", None),
                        "filename": h_name,
                        "gemini_key": get_active_gemini_api_key(),
                    }
                    st.rerun()

        # CSV Export for Session History with Local and UTC timestamps
        hist_df = pd.DataFrame([
            {
                "Filename": h.get("filename"),
                "Duration (s)": h.get("duration_sec"),
                "Synthetic Ratio (%)": h.get("fake_ratio"),
                "Verdict": h.get("verdict"),
                "Timestamp (Local)": format_local_timestamp(h.get("created_at_utc"), user_tz_obj, user_tz_name) if h.get("created_at_utc") else h.get("created_at"),
                "Timezone": user_tz_name,
                "Timestamp (UTC)": h.get("created_at_utc", ""),
            }
            for h in filtered_scans
        ])
        hist_csv = io.StringIO()
        hist_df.to_csv(hist_csv, index=False)
        st.download_button(
            label=f"📥 Export History Records (CSV) — [{user_tz_name}]",
            data=hist_csv.getvalue(),
            file_name="AudioArtifact_Session_History.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div style="background:#131614;border:1px dashed #262B27;border-radius:10px;padding:28px;text-align:center;color:#8C958E;font-size:13.5px;">
          No matching forensic scans found. Upload and analyze an audio clip in the 'Single Audio Localizer' tab to populate this audit log.
        </div>
        """, unsafe_allow_html=True)
