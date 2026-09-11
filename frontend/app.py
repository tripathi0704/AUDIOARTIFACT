"""
app.py — AudioArtifact v2.0 Forensic Dashboard
------------------------------------------------
Timeline-Based Deepfake Audio Localizer Interface.
Features:
- Drag-and-drop audio uploader (.wav / .mp3)
- Instant playback & analysis trigger
- Forensic Verdict Card (Fake ratio %, Highest risk segment, Average confidence)
- Plotly Continuous Deepfake Probability Curve across time with 50% risk threshold
- Interactive Forensic Segment Explorer (Click-to-inspect timestamps & risks)
- 3D Voice Signature Spectrogram (Mel-frequency topography)
- SQLite Scan History Inspector
"""

import os
import io
import json
import requests
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import librosa

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AudioArtifact v2.0 — Deepfake Audio Localizer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------
# GLOBAL DARK THEME STYLING
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg: #0A0C0B;
  --panel: #131614;
  --panel-2: #1A1E1B;
  --line: #262B27;
  --text: #ECEFEB;
  --muted: #8C958E;
  --muted-dim: #5E6660;
  --real: #3ECF8E;
  --real-dim: rgba(62,207,142,0.12);
  --fake: #FF5C5C;
  --fake-dim: rgba(255,92,92,0.14);
  --amber: #FFB454;
  --amber-dim: rgba(255,180,84,0.14);
  --cyan: #5EEAD4;
}

#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
  visibility: hidden;
  height: 0;
}

.stApp {
  background: var(--bg);
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1120px;
}

html, body, [class*="css"] {
  font-family: 'Manrope', sans-serif;
  color: var(--text);
}

.mono {
  font-family: 'IBM Plex Mono', monospace;
}

/* Brand Header */
.brand-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--line);
}

.brand {
  font-family: 'IBM Plex Mono', monospace;
  font-weight: 700;
  font-size: 17px;
  display: flex;
  align-items: center;
  gap: 10px;
  letter-spacing: 0.02em;
}

.brand .px {
  width: 9px;
  height: 9px;
  background: var(--real);
  border-radius: 50%;
  box-shadow: 0 0 10px var(--real);
  animation: pulse 2.2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

.status-badge {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 5px 12px;
  border-radius: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-badge i {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--real);
  box-shadow: 0 0 6px var(--real);
}

.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 800;
  letter-spacing: -0.02em;
  font-size: clamp(30px, 4.2vw, 46px);
  line-height: 1.15;
  margin-bottom: 8px;
}

.hero-title span.fake { color: var(--fake); }
.hero-title span.real { color: var(--real); }

.hero-sub {
  color: var(--muted);
  font-size: 15.5px;
  max-width: 680px;
  margin-bottom: 28px;
  line-height: 1.6;
}

/* File Uploader Restyle */
[data-testid="stFileUploader"] {
  background: var(--panel);
  border: 1.5px dashed var(--line);
  border-radius: 14px;
  padding: 10px 14px;
  transition: border-color 0.25s ease;
}

[data-testid="stFileUploader"]:hover {
  border-color: var(--real);
}

[data-testid="stFileUploader"] section {
  background: transparent;
  border: none;
}

/* Buttons */
div.stButton > button {
  background: var(--text);
  color: #0A0C0B;
  border: none;
  font-weight: 700;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 14px;
  padding: 10px 24px;
  border-radius: 22px;
  transition: transform 0.15s ease, background 0.15s ease;
}

div.stButton > button:hover {
  transform: translateY(-1px);
  background: #ffffff;
}

/* Section Labels */
.section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--amber);
  margin: 32px 0 14px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.section-label::before {
  content: '';
  width: 20px;
  height: 1px;
  background: var(--amber);
}

/* Verdict Cards */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
  margin-bottom: 24px;
}

.vcard {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 20px;
  transition: border-color 0.2s;
}

.vcard .k {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--muted-dim);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
}

.vcard .v {
  font-size: 24px;
  font-weight: 700;
  font-family: 'Space Grotesk', sans-serif;
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
  padding: 8px 16px;
  border-radius: 20px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  font-weight: 600;
}

.pill-status.real {
  background: var(--real-dim);
  border: 1px solid rgba(62,207,142,0.35);
  color: var(--real);
}

.pill-status.fake {
  background: var(--fake-dim);
  border: 1px solid rgba(255,92,92,0.35);
  color: var(--fake);
}

.pill-status.mixed {
  background: var(--amber-dim);
  border: 1px solid rgba(255,180,84,0.35);
  color: var(--amber);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HEADER & BRAND
# ----------------------------------------------------------------------
st.markdown("""
<div class="brand-nav">
  <div class="brand"><span class="px"></span>AudioArtifact <span style="font-size:12px;color:#8C958E;font-weight:400;">v2.0</span></div>
  <div class="status-badge"><i></i>ENGINE: VAD + 50% OVERLAP + 60-MFCC</div>
</div>
<h1 class="hero-title">Audio<span class="real">Artifact</span> — Timeline-Based <span class="fake">Deepfake</span> Audio Localizer</h1>
<p class="hero-sub">
  Forensic-grade audio analysis utilizing Silero VAD, 50% overlapping windows (2s window, 1s stride),
  and 60-feature dynamic derivatives (MFCC + Delta + Delta-Delta) with continuous probability reporting.
</p>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# FILE UPLOAD & ACTIONS
# ----------------------------------------------------------------------
col_up, col_ctrl = st.columns([3, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "Upload suspicious audio clip",
        type=["mp3", "wav"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    st.audio(uploaded_file)
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        analyze_clicked = st.button("◈ Run Forensic Analysis", type="primary", use_container_width=True)
else:
    analyze_clicked = False

# ----------------------------------------------------------------------
# ANALYSIS EXECUTION
# ----------------------------------------------------------------------
if uploaded_file is not None and analyze_clicked:
    with st.spinner("Processing audio: Running Silero VAD, 50% overlapping windowing & 60-feature inference..."):
        uploaded_file.seek(0)
        file_bytes = uploaded_file.getvalue()
        data = None

        # 1. Try FastAPI backend first (when running multi-server or Docker)
        try:
            files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
            res = requests.post(f"{BACKEND_URL}/analyze", files=files, timeout=120)
            if res.status_code == 200:
                data = res.json()
        except Exception:
            pass

        # 2. Seamless fallback to internal engine (for Streamlit Community Cloud)
        if not data or "error" in data:
            try:
                import sys
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from backend.main import analyze_audio_data
                data = analyze_audio_data(file_bytes, uploaded_file.name)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

    if "error" in data:
        st.error(data["error"])
        st.stop()

    st.session_state["last_result"] = data
    st.session_state["analyzed_file_name"] = uploaded_file.name

# ----------------------------------------------------------------------
# RENDER FORENSIC REPORT
# ----------------------------------------------------------------------
if "last_result" in st.session_state:
    data = st.session_state["last_result"]
    segments = data["segments"]
    total_dur = data.get("total_duration", 0.0)
    fake_ratio = data.get("fake_ratio", 0.0)
    verdict = data.get("verdict", "")
    highest = data.get("highest_risk_segment", {})
    avg_conf = round(sum(s["confidence"] for s in segments) / max(len(segments), 1), 1)

    # Verdict Pill Style
    if fake_ratio == 0:
        pill_class = "real"
        pill_text = f"✓ {verdict}"
    elif fake_ratio >= 80:
        pill_class = "fake"
        pill_text = f"⚠ {verdict}"
    else:
        pill_class = "mixed"
        pill_text = f"⚡ {verdict}"

    st.markdown('<div class="section-label">Forensic Summary</div>', unsafe_allow_html=True)

    # Top Status & Metrics Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="vcard">
          <div class="k">Overall Verdict</div>
          <div style="margin-top:6px;"><span class="pill-status {pill_class}">{pill_text}</span></div>
          <div class="sub">File: {data.get("filename", "")}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="vcard">
          <div class="k">Synthetic Audio Ratio</div>
          <div class="v" style="color:{'#FF5C5C' if fake_ratio > 40 else '#3ECF8E'}">{fake_ratio}%</div>
          <div class="sub">{data.get("fake_seconds", 0)}s of {total_dur}s flagged</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        peak_prob = round(highest.get("fake_probability", 0.0) * 100, 1)
        st.markdown(f"""
        <div class="vcard">
          <div class="k">Peak Risk Segment</div>
          <div class="v" style="color:{'#FF5C5C' if peak_prob >= 50 else '#3ECF8E'}">{peak_prob}% AI</div>
          <div class="sub">Timestamp: {highest.get('start', 0)}s – {highest.get('end', 0)}s</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="vcard">
          <div class="k">Model Confidence</div>
          <div class="v" style="color:var(--cyan);">{avg_conf}%</div>
          <div class="sub">{len(segments)} overlapping windows</div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # CONTINUOUS PROBABILITY TIMELINE (PLOTLY)
    # ------------------------------------------------------------------
    st.markdown('<div class="section-label">Continuous Deepfake Probability Curve (Timeline)</div>', unsafe_allow_html=True)

    time_points = [round((s["start"] + s["end"]) / 2, 2) for s in segments]
    fake_probs = [round(s["fake_probability"] * 100, 2) for s in segments]
    confidences = [s["confidence"] for s in segments]
    hover_texts = [
        f"<b>Window {s['segment']+1}</b><br>"
        f"Interval: {s['start']}s – {s['end']}s<br>"
        f"AI Probability: {s['fake_probability']*100:.1f}%<br>"
        f"Classification: <b>{s['label']}</b> (Confidence: {s['confidence']}%)"
        for s in segments
    ]

    fig = go.Figure()

    # 50% Threshold Area
    fig.add_shape(
        type="rect",
        x0=0, x1=total_dur,
        y0=50, y1=100,
        fillcolor="rgba(255, 92, 92, 0.05)",
        line=dict(width=0),
        layer="below"
    )

    # 50% Threshold Line
    fig.add_trace(go.Scatter(
        x=[0, total_dur],
        y=[50, 50],
        mode="lines",
        line=dict(color="rgba(255, 180, 84, 0.7)", width=1.5, dash="dash"),
        name="AI Decision Boundary (50%)",
        hoverinfo="skip"
    ))

    # Continuous Probability Line
    fig.add_trace(go.Scatter(
        x=time_points,
        y=fake_probs,
        mode="lines+markers",
        name="Deepfake Probability",
        line=dict(color="#3ECF8E", width=3, shape="spline"),
        marker=dict(
            size=7,
            color=["#FF5C5C" if p >= 50 else "#3ECF8E" for p in fake_probs],
            line=dict(color="#0A0C0B", width=1.5)
        ),
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#131614",
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(
            title=dict(text="Timeline (Seconds)", font=dict(family="IBM Plex Mono", size=11, color="#8C958E")),
            tickfont=dict(family="IBM Plex Mono", size=10, color="#5E6660"),
            gridcolor="#262B27",
            zeroline=False,
            range=[0, total_dur]
        ),
        yaxis=dict(
            title=dict(text="AI Voice Probability (%)", font=dict(family="IBM Plex Mono", size=11, color="#8C958E")),
            tickfont=dict(family="IBM Plex Mono", size=10, color="#5E6660"),
            gridcolor="#262B27",
            range=[-2, 105],
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="IBM Plex Mono", size=11, color="#8C958E")
        ),
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ------------------------------------------------------------------
    # SEGMENT-BY-SEGMENT FORENSIC EXPLORER
    # ------------------------------------------------------------------
    st.markdown('<div class="section-label">Overlapping Window Breakdown</div>', unsafe_allow_html=True)

    with st.expander(f"Inspect all {len(segments)} windows (2.0s window / 1.0s stride)", expanded=False):
        seg_cols = st.columns(4)
        for idx, s in enumerate(segments):
            with seg_cols[idx % 4]:
                is_fake = s["label_code"] == 1
                color = "#FF5C5C" if is_fake else "#3ECF8E"
                st.markdown(f"""
                <div style="background:#1A1E1B;border:1px solid #262B27;border-left:3px solid {color};padding:10px 12px;border-radius:8px;margin-bottom:10px;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#8C958E;">
                    WINDOW {s['segment']+1} · {s['start']}s–{s['end']}s
                  </div>
                  <div style="font-weight:700;font-size:14px;color:{color};margin-top:2px;">
                    {s['label']} ({s['fake_probability']*100:.1f}%)
                  </div>
                  <div style="font-size:11px;color:#5E6660;">Confidence: {s['confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3D VOICE SIGNATURE SPECTROGRAM
    # ------------------------------------------------------------------
    st.markdown('<div class="section-label">Voice Signature Spectrogram</div>', unsafe_allow_html=True)

    spec_data = data.get("spectrogram", {})
    if "z" in spec_data and spec_data["z"]:
        z = np.array(spec_data["z"])
        x = np.array(spec_data.get("time", list(range(z.shape[1]))))
        y = np.array(spec_data.get("mel", list(range(z.shape[0]))))

        fig3d = go.Figure(data=[go.Surface(
            z=z, x=x, y=y,
            colorscale=[
                [0, "#0A0C0B"],
                [0.35, "#132E22"],
                [0.7, "#1F6B4C"],
                [1.0, "#3ECF8E"]
            ],
            showscale=False
        )])
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title="", showbackground=False, color="#5E6660"),
                yaxis=dict(title="", showbackground=False, color="#5E6660"),
                zaxis=dict(title="", showbackground=False, color="#5E6660"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=380,
        )
        st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------
# SCAN HISTORY (SQLITE)
# ----------------------------------------------------------------------
st.markdown('<div class="section-label">Database Scan History</div>', unsafe_allow_html=True)
try:
    hist_res = requests.get(f"{BACKEND_URL}/history?limit=10", timeout=5)
    if hist_res.status_code == 200:
        history_records = hist_res.json()
        if history_records:
            with st.expander(f"Recent Database Records ({len(history_records)} scans)", expanded=False):
                for h in history_records:
                    v = h.get("verdict", "")
                    col_v = "#3ECF8E" if "Authentic" in v or "Human" in v else "#FF5C5C" if "Synthetic" in v or "Fake" in v else "#FFB454"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;background:#131614;border:1px solid #262B27;border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                      <div>
                        <div style="font-weight:600;font-size:13.5px;">{h.get('filename')}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8C958E;margin-top:2px;">
                          Duration: {h.get('duration_sec', 0):.1f}s · Created: {h.get('created_at', '')[:19].replace('T', ' ')}
                        </div>
                      </div>
                      <div style="text-align:right;">
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:11.5px;font-weight:600;color:{col_v};background:#1A1E1B;padding:4px 10px;border-radius:12px;">
                          {v or 'Scanned'}
                        </span>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8C958E;margin-top:3px;">
                          Fake Ratio: {h.get('fake_ratio', 0):.1f}%
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("No scan records in history database yet.")
except Exception:
    st.caption("Backend offline or history currently unavailable.")
