"""
app.py — AudioArtifact ARF-200 Forensic Dashboard (90s Rackmount Edition)
-------------------------------------------------------------------------
Timeline-Based Deepfake Audio Localizer Interface styled after authentic
1990s studio rackmount audio forensic hardware (ARF-200 LOCALIZER).

Features:
- Dual backlit analog VU meters (REAL vs FAKE with warm tungsten incandescent dials)
- Green phosphor CRT / VFD oscilloscope screen with scanlines & telemetry
- Tactile 90s hardware push-buttons with amber backlight indicators
- Industrial brushed aluminum & dark steel rackmount chassis with hex screws
- Silero VAD silence filtering + 50% overlapping windows (2.0s window, 1.0s stride)
- Microsoft WavLM 768-dimensional deep acoustic embeddings
- Continuous probability timeline with 50% risk threshold
- Overlapping window channel strip breakdown & 3D sonograph spectrogram
- Isolated, ephemeral browser session history
"""

import os
import sys
import types
import io
import json
import uuid
import requests

# Windows 11 Smart App Control & cross-platform safe numba bypass for librosa
try:
    from numba import _dispatcher
except Exception:
    _m = types.ModuleType("numba")
    _m.jit = lambda *a, **kw: (lambda f: f) if a and callable(a[0]) else (lambda f: f)
    _m.njit = _m.jit
    _m.vectorize = _m.jit
    _m.guvectorize = _m.jit
    _m.stencil = _m.jit
    _m.prange = range
    sys.modules["numba"] = _m
    sys.modules["numba.core"] = types.ModuleType("numba.core")
    sys.modules["numba.core.decorators"] = _m

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import librosa

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def is_local_backend_active(port: int = 8000) -> bool:
    """Non-blocking 0.05s check to see if local uvicorn is actually listening."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.05)
        s.connect(("127.0.0.1", port))
        s.close()
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner="Initializing ARF-200 foundation models (WavLM CPU weights)...")
def preload_cloud_models():
    """Download and cache foundation models into Streamlit Cloud memory on startup."""
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from backend.main import get_model, get_wavlm
    get_model()
    return get_wavlm()


# If local FastAPI backend on port 8000 is NOT active, preload and cache once
if not is_local_backend_active(8000):
    preload_cloud_models()

st.set_page_config(
    page_title="ARF-200 — AudioArtifact Deepfake Localizer",
    page_icon="📻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------
# EPHEMERAL SESSION INITIALIZATION (ISOLATED PER BROWSER WINDOW)
# ----------------------------------------------------------------------
if "session_token" not in st.session_state:
    st.session_state["session_token"] = uuid.uuid4().hex[:6].upper()
if "session_scans" not in st.session_state:
    st.session_state["session_scans"] = []


# ----------------------------------------------------------------------
# ANALOG VU METER SVG GENERATOR (WARM TUNGSTEN INCANDESCENT BACKLIGHT)
# ----------------------------------------------------------------------
def generate_analog_vu_meter_svg(title: str, value: float, is_fake_meter: bool = False) -> str:
    """
    Renders an authentic 1990s backlit analog VU meter with glowing amber tungsten dial,
    curved dB scale, red peak overload zone, and physically calculated needle deflection.
    value: 0.0 to 100.0
    """
    val = max(0.0, min(100.0, float(value)))
    # Map 0..100% to -44 deg (far left -30dB) up to +44 deg (far right +6dB)
    angle = -44.0 + (val / 100.0 * 88.0)

    meter_id = "vu_fake" if is_fake_meter else "vu_real"
    label_text = "FAKE" if is_fake_meter else "REAL"
    sub_label = "SYNTHETIC THREAT" if is_fake_meter else "HUMAN FIDELITY"

    svg = f"""
    <svg viewBox="0 0 250 146" class="analog-vu-svg" style="width:100%;max-width:245px;height:auto;display:block;margin:0 auto;filter:drop-shadow(0 6px 14px rgba(0,0,0,0.85));">
      <defs>
        <!-- Warm tungsten incandescent backlight -->
        <radialGradient id="grad_{meter_id}" cx="50%" cy="85%" r="80%">
          <stop offset="0%" stop-color="#fff5d4"/>
          <stop offset="35%" stop-color="#fdbb5e"/>
          <stop offset="70%" stop-color="#c67d1d"/>
          <stop offset="100%" stop-color="#462505"/>
        </radialGradient>

        <!-- Bezel frame gradient -->
        <linearGradient id="bezel_{meter_id}" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#60666d"/>
          <stop offset="20%" stop-color="#2a2e33"/>
          <stop offset="80%" stop-color="#191b1e"/>
          <stop offset="100%" stop-color="#3d434a"/>
        </linearGradient>

        <!-- Glass reflection filter -->
        <linearGradient id="glass_{meter_id}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="rgba(255,255,255,0.28)"/>
          <stop offset="30%" stop-color="rgba(255,255,255,0.06)"/>
          <stop offset="60%" stop-color="rgba(255,255,255,0)"/>
        </linearGradient>
      </defs>

      <!-- Outer Metal Bezel -->
      <rect x="2" y="2" width="246" height="142" rx="9" ry="9" fill="url(#bezel_{meter_id})" stroke="#0c0d0f" stroke-width="2"/>
      <!-- Inner Bezel Ring -->
      <rect x="7" y="7" width="236" height="132" rx="6" ry="6" fill="#131518" stroke="#000000" stroke-width="1.8"/>

      <!-- Backlit Dial Face -->
      <rect x="10" y="10" width="230" height="126" rx="4" ry="4" fill="url(#grad_{meter_id})"/>

      <!-- Corner Bezel Screws -->
      <circle cx="8" cy="8" r="2.2" fill="#889099" stroke="#222"/>
      <circle cx="242" cy="8" r="2.2" fill="#889099" stroke="#222"/>
      <circle cx="8" cy="138" r="2.2" fill="#889099" stroke="#222"/>
      <circle cx="242" cy="138" r="2.2" fill="#889099" stroke="#222"/>

      <!-- Scale Arcs -->
      <!-- Black Safe Zone Arc -->
      <path d="M 38 88 A 108 108 0 0 1 178 44" fill="none" stroke="#1c160f" stroke-width="3" stroke-linecap="round"/>
      <!-- Red Danger Zone Arc -->
      <path d="M 178 44 A 108 108 0 0 1 214 74" fill="none" stroke="#d62424" stroke-width="4" stroke-linecap="round"/>

      <!-- Scale Ticks & Legend -->
      <line x1="40" y1="86" x2="46" y2="91" stroke="#221a11" stroke-width="1.8"/>
      <text x="35" y="102" font-family="'Share Tech Mono', monospace" font-size="8" font-weight="700" fill="#291e12" text-anchor="middle">-30</text>

      <line x1="66" y1="66" x2="71" y2="71" stroke="#221a11" stroke-width="1.6"/>
      <text x="64" y="59" font-family="'Share Tech Mono', monospace" font-size="7.5" font-weight="700" fill="#291e12" text-anchor="middle">-10</text>

      <line x1="98" y1="51" x2="101" y2="57" stroke="#221a11" stroke-width="1.6"/>
      <text x="97" y="44" font-family="'Share Tech Mono', monospace" font-size="7.5" font-weight="700" fill="#291e12" text-anchor="middle">-5</text>

      <line x1="124" y1="45" x2="125" y2="52" stroke="#221a11" stroke-width="1.8"/>
      <text x="124" y="38" font-family="'Share Tech Mono', monospace" font-size="8" font-weight="700" fill="#291e12" text-anchor="middle">-3</text>

      <line x1="150" y1="44" x2="149" y2="51" stroke="#221a11" stroke-width="1.6"/>
      <text x="151" y="37" font-family="'Share Tech Mono', monospace" font-size="7.5" font-weight="700" fill="#291e12" text-anchor="middle">-1</text>

      <!-- 0 dB (Threshold) -->
      <line x1="177" y1="45" x2="175" y2="53" stroke="#d62424" stroke-width="2.2"/>
      <text x="176" y="37" font-family="'Share Tech Mono', monospace" font-size="8.5" font-weight="800" fill="#a81111" text-anchor="middle">0</text>

      <!-- +2 dB -->
      <line x1="197" y1="57" x2="192" y2="63" stroke="#d62424" stroke-width="1.8"/>
      <text x="202" y="52" font-family="'Share Tech Mono', monospace" font-size="7.5" font-weight="700" fill="#a81111" text-anchor="middle">+2</text>

      <!-- +6 dB -->
      <line x1="212" y1="72" x2="205" y2="77" stroke="#d62424" stroke-width="2.2"/>
      <text x="220" y="70" font-family="'Share Tech Mono', monospace" font-size="8" font-weight="800" fill="#a81111" text-anchor="middle">+6</text>

      <!-- Unit Badge -->
      <text x="208" y="102" font-family="'Share Tech Mono', monospace" font-size="7.5" font-weight="700" fill="#4d3215">dB VU</text>

      <!-- Main Meter Label (REAL or FAKE) -->
      <text x="125" y="105" font-family="'Space Grotesk', sans-serif" font-size="13.5" font-weight="900" letter-spacing="1.5" fill="#1c1409" text-anchor="middle">{label_text}</text>
      <text x="125" y="118" font-family="'Share Tech Mono', monospace" font-size="7" letter-spacing="0.8" fill="#543719" text-anchor="middle">{sub_label} · {val:.0f}%</text>

      <!-- Calibrated Needle -->
      <g transform="rotate({angle:.1f}, 125, 138)" style="transition: transform 0.8s cubic-bezier(0.18, 0.89, 0.32, 1.28);">
        <!-- Needle Shadow -->
        <polygon points="123,138 127,138 125.6,26 124.4,26" fill="rgba(60, 30, 5, 0.4)" transform="translate(3, 3)"/>
        <!-- Needle Body -->
        <polygon points="123.8,138 126.2,138 125.4,24 124.6,24" fill="#0f0c08"/>
        <!-- Needle Tip in Danger Red if in Overload -->
        <line x1="125" y1="24" x2="125" y2="38" stroke="{'#e62e2e' if angle > 8 else '#1a140d'}" stroke-width="1.8"/>
      </g>

      <!-- Pivot Cap (Vintage Bakelite style) -->
      <circle cx="125" cy="138" r="15" fill="#181a1d" stroke="#000000" stroke-width="2"/>
      <circle cx="125" cy="138" r="12" fill="#2d3138"/>
      <circle cx="125" cy="138" r="5.5" fill="#0d0e10"/>
      <circle cx="122" cy="135" r="1.8" fill="rgba(255,255,255,0.4)"/>

      <!-- Glass Glare Highlight -->
      <rect x="10" y="10" width="230" height="60" fill="url(#glass_{meter_id})" rx="4" pointer-events="none"/>
    </svg>
    """
    return svg


# ----------------------------------------------------------------------
# 1990s HARDWARE RACKMOUNT INDUSTRIAL DESIGN SYSTEM (CSS)
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&family=Space+Grotesk:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

:root{
  --rack-metal: #23272d;
  --rack-metal-light: #373e46;
  --rack-metal-dark: #16191c;
  --rack-bezel: #111316;
  --crt-green: #39ff14;
  --crt-green-dim: #1d7e0c;
  --crt-bg: #031206;
  --amber-lamp: #ffb454;
  --amber-glow: rgba(255, 180, 84, 0.45);
  --danger-red: #ff3838;
  --danger-glow: rgba(255, 56, 56, 0.45);
  --cyan-led: #5eead4;
  --screws: #69737d;
}

#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
  visibility: hidden;
  height: 0;
}

.stApp {
  background: #0d0f12;
  background-image: 
    radial-gradient(ellipse at 50% 0%, #1a2027 0%, #0d0f12 75%),
    repeating-linear-gradient(45deg, rgba(0,0,0,0.15) 0, rgba(0,0,0,0.15) 2px, transparent 2px, transparent 4px);
  color: #dbe2ea;
  font-family: 'IBM Plex Mono', monospace;
}

.block-container {
  padding-top: 1.5rem;
  padding-bottom: 3.5rem;
  max-width: 1140px;
}

/* 19" Rackmount Chassis Frame */
.rack-outer {
  background: linear-gradient(180deg, #30363e 0%, #20242a 20%, #191c21 80%, #292f37 100%);
  border: 2px solid #3f4752;
  border-radius: 12px;
  box-shadow: 
    0 18px 45px rgba(0, 0, 0, 0.95),
    inset 0 1px 0 rgba(255,255,255,0.2),
    inset 0 -2px 0 rgba(0,0,0,0.8);
  padding: 18px 24px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}

/* Side Rack Mounting Ears with Hex Screws */
.rack-ear-left, .rack-ear-right {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 22px;
  background: linear-gradient(90deg, #181b1f 0%, #2c3239 50%, #15181b 100%);
  border-right: 1.5px solid #0d0f12;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0;
  z-index: 5;
}
.rack-ear-left { left: 0; border-right: 2px solid #111316; }
.rack-ear-right { right: 0; border-left: 2px solid #111316; }

.screw {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #9aa3ad 0%, #464d57 70%, #1f2328 100%);
  border: 1px solid #15181b;
  box-shadow: inset 0 1px 1px rgba(255,255,255,0.5), 0 1px 3px rgba(0,0,0,0.8);
  position: relative;
}
.screw::after {
  content: '';
  position: absolute;
  top: 4px; left: 1.5px; width: 7px; height: 1.5px;
  background: #111417;
  transform: rotate(35deg);
}

.rack-title-vertical {
  writing-mode: vertical-rl;
  text-orientation: mixed;
  transform: rotate(180deg);
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 800;
  font-size: 9px;
  letter-spacing: 0.22em;
  color: #65707d;
  text-shadow: 0 1px 0 rgba(255,255,255,0.1);
}

/* Master Hardware Header */
.rack-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 14px 14px 14px;
  border-bottom: 1.5px solid #14171a;
  box-shadow: 0 1px 0 rgba(255,255,255,0.06);
  margin-bottom: 16px;
}

.rack-brand {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 20px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #e3e8ef;
  display: flex;
  align-items: center;
  gap: 12px;
}
.rack-brand span.badge-num {
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px;
  background: #121518;
  border: 1px solid #373e46;
  color: #5eead4;
  padding: 3px 8px;
  border-radius: 4px;
  letter-spacing: 0.1em;
}

.telemetry-tag {
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px;
  color: #8f9aa6;
  display: flex;
  align-items: center;
  gap: 14px;
}

/* Vintage Rocker Switch & LEDs */
.rocker-unit {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: #14171b;
  border: 1px solid #2d333b;
  padding: 4px 10px;
  border-radius: 5px;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);
}

.neon-lamp {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #ffb454;
  box-shadow: 0 0 8px #ffb454, 0 0 16px rgba(255,180,84,0.6);
  border: 1px solid #fff;
  display: inline-block;
  animation: neonGlow 2.5s infinite alternate ease-in-out;
}
@keyframes neonGlow {
  0% { opacity: 0.85; filter: brightness(1); }
  100% { opacity: 1.0; filter: brightness(1.25); }
}

.led-indicator {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  display: inline-block;
  border: 1px solid rgba(0,0,0,0.6);
}
.led-indicator.green {
  background: #39ff14;
  box-shadow: 0 0 8px #39ff14, 0 0 14px rgba(57,255,20,0.5);
}
.led-indicator.red {
  background: #ff3838;
  box-shadow: 0 0 8px #ff3838, 0 0 14px rgba(255,56,56,0.6);
  animation: clipPulse 1.2s infinite;
}
.led-indicator.dim {
  background: #381919;
  box-shadow: none;
}
@keyframes clipPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

/* Green Phosphor CRT Display Frame */
.crt-bezel {
  background: #14181c;
  border: 3px solid #262c33;
  border-radius: 14px;
  padding: 8px;
  box-shadow: 
    inset 0 4px 12px rgba(0,0,0,0.95),
    0 4px 10px rgba(0,0,0,0.6);
  margin-bottom: 20px;
  position: relative;
}

.crt-screen-inner {
  background: #031206;
  border: 2px solid #0e3016;
  border-radius: 8px;
  padding: 12px 16px;
  position: relative;
  overflow: hidden;
  box-shadow: inset 0 0 45px rgba(20, 110, 35, 0.28);
}

/* Authentic CRT Scanline texture overlay */
.crt-screen-inner::before {
  content: " ";
  display: block;
  position: absolute;
  top: 0; left: 0; bottom: 0; right: 0;
  background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.35) 50%), linear-gradient(90deg, rgba(30, 255, 30, 0.02), rgba(0, 255, 0, 0.01), rgba(0, 50, 255, 0.02));
  z-index: 2;
  background-size: 100% 3px, 6px 100%;
  pointer-events: none;
}

.crt-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: 'VT323', monospace;
  font-size: 19px;
  letter-spacing: 0.14em;
  color: #39ff14;
  text-shadow: 0 0 6px rgba(57, 255, 20, 0.65);
  margin-bottom: 4px;
  border-bottom: 1px dashed rgba(57, 255, 20, 0.25);
  padding-bottom: 6px;
}

.crt-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: 'Share Tech Mono', monospace;
  font-size: 10.5px;
  color: rgba(57, 255, 20, 0.75);
  text-shadow: 0 0 4px rgba(57, 255, 20, 0.4);
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px dashed rgba(57, 255, 20, 0.25);
}

/* 90s Mechanical Push-Buttons */
div.stButton > button {
  background: linear-gradient(180deg, #373d45 0%, #252a30 45%, #191c20 100%) !important;
  color: #f0f3f6 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 13.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border: 1px solid #4a525d !important;
  border-radius: 6px !important;
  padding: 10px 22px !important;
  box-shadow: 
    0 4px 0 #0d0f12,
    0 6px 10px rgba(0,0,0,0.6),
    inset 0 1px 0 rgba(255,255,255,0.25) !important;
  transition: transform 0.08s ease, box-shadow 0.08s ease !important;
  position: relative !important;
}

div.stButton > button:hover {
  background: linear-gradient(180deg, #444c56 0%, #2c323a 45%, #1e2226 100%) !important;
  border-color: #ffb454 !important;
  color: #ffffff !important;
  transform: translateY(1px) !important;
  box-shadow: 
    0 3px 0 #0d0f12,
    0 4px 8px rgba(0,0,0,0.6),
    inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

div.stButton > button:active {
  transform: translateY(4px) !important;
  box-shadow: 
    0 0 0 #0d0f12,
    inset 0 2px 4px rgba(0,0,0,0.7) !important;
}

/* File Uploader 90s Audio Cartridge Input Bay */
[data-testid="stFileUploader"] {
  background: #15181c !important;
  border: 2px dashed #3a424c !important;
  border-radius: 8px !important;
  padding: 12px 16px !important;
  box-shadow: inset 0 2px 6px rgba(0,0,0,0.7) !important;
  transition: border-color 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: #5eead4 !important;
}

/* Section Header Labels */
.hardware-section-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #ffb454;
  margin: 24px 0 12px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.hardware-section-label::before {
  content: '';
  width: 14px;
  height: 2px;
  background: #ffb454;
  box-shadow: 0 0 6px rgba(255, 180, 84, 0.6);
}

/* VFD Fluorescent Digital Readout Panels */
.readout-card {
  background: #121519;
  border: 1.5px solid #282f37;
  border-radius: 8px;
  padding: 14px 16px;
  box-shadow: inset 0 2px 8px rgba(0,0,0,0.8), 0 2px 5px rgba(0,0,0,0.4);
}
.readout-k {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  color: #7b8694;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.readout-v {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.readout-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10.5px;
  color: #636d7a;
  margin-top: 3px;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# MASTER RACKMOUNT CHASSIS WRAPPER (TOP SECTION)
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="rack-outer">
  <div class="rack-ear-left">
    <div class="screw"></div>
    <div class="rack-title-vertical">ARF-200</div>
    <div class="screw"></div>
  </div>
  <div class="rack-ear-right">
    <div class="screw"></div>
    <div class="rack-title-vertical">LOCALIZER</div>
    <div class="screw"></div>
  </div>

  <!-- Header Bar -->
  <div class="rack-header-row">
    <div class="rack-brand">
      <span>AUDIO ARTIFACT</span>
      <span class="badge-num">ARF-200 · SOTA</span>
    </div>
    <div class="telemetry-tag">
      <div class="rocker-unit">
        <span class="neon-lamp"></span>
        <span style="font-weight:700;color:#ffb454;">POWER [ON]</span>
      </div>
      <div class="rocker-unit">
        <span class="led-indicator green"></span>
        <span>VAD ACTIVE</span>
      </div>
      <div class="rocker-unit">
        <span style="color:#5eead4;">SESSION #{st.session_state['session_token']}</span>
      </div>
    </div>
  </div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# DUAL ANALOG VU METERS & HARDWARE TELEMETRY PANEL
# ----------------------------------------------------------------------
# Check if analysis has already run in this session
last_data = st.session_state.get("last_result", None)

if last_data:
    fake_ratio_val = float(last_data.get("fake_ratio", 0.0))
    highest_info = last_data.get("highest_risk_segment", {})
    peak_fake_prob = float(highest_info.get("fake_probability", fake_ratio_val / 100.0)) * 100.0
    human_conf_val = max(0.0, 100.0 - fake_ratio_val)
    has_clip = peak_fake_prob >= 50.0
    status_msg = "DETECTING ARTIFACTS [FLAGGED]" if has_clip else "SYSTEM VERIFIED [AUTHENTIC]"
else:
    fake_ratio_val = 0.0
    peak_fake_prob = 0.0
    human_conf_val = 100.0
    has_clip = False
    status_msg = "SYSTEM STANDBY · AWAITING AUDIO CARTRIDGE"

col_meter_ctrl, col_meter_left, col_meter_right = st.columns([1.3, 1, 1])

with col_meter_ctrl:
    clip_led_class = "red" if has_clip else "dim"
    st.markdown(f"""
    <div style="background:#14171b;border:1.5px solid #292f37;border-radius:8px;padding:12px 14px;height:146px;box-shadow:inset 0 2px 6px rgba(0,0,0,0.7);display:flex;flex-direction:column;justify-content:space-between;">
      <div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#ffb454;letter-spacing:0.1em;font-weight:700;">
          TELEMETRY // RACK STATUS
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#8f9aa6;margin-top:4px;line-height:1.5;">
          SAMPLE RATE: 16.0kHz / 16-BIT<br>
          ENGINE: SILERO VAD + WAVLM 768D<br>
          WINDOW: 2.0s · STRIDE: 1.0s (50% OVERLAP)
        </div>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;border-top:1px dashed #282f37;padding-top:6px;">
        <div style="display:flex;align-items:center;gap:6px;">
          <span class="led-indicator {clip_led_class}"></span>
          <span style="font-family:'Share Tech Mono',monospace;font-size:10px;font-weight:700;color:{'#ff3838' if has_clip else '#636d7a'};">
            CLIP / THREAT {'[ACTIVE]' if has_clip else '[CLEAR]'}
          </span>
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:9.5px;color:#5eead4;">
          ARF-200
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_meter_left:
    st.markdown(generate_analog_vu_meter_svg("REAL", human_conf_val, is_fake_meter=False), unsafe_allow_html=True)

with col_meter_right:
    st.markdown(generate_analog_vu_meter_svg("FAKE", peak_fake_prob, is_fake_meter=True), unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 90s AUDIO INPUT BAY (FILE UPLOAD & PLAYBACK)
# ----------------------------------------------------------------------
st.markdown('<div class="hardware-section-label">INPUT MEDIA BAY // 1/4" AUDIO CASSETTE TRAY</div>', unsafe_allow_html=True)

col_input, col_actions = st.columns([2.8, 1.2])

with col_input:
    uploaded_file = st.file_uploader(
        "Upload audio cartridge (.wav or .mp3)",
        type=["mp3", "wav"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    st.audio(uploaded_file)
    with col_actions:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("◈ ANALYZE CARTRIDGE", type="primary", use_container_width=True)
else:
    analyze_clicked = False
    with col_actions:
        st.markdown("""
        <div style="background:#13161a;border:1px dashed #2b323a;border-radius:6px;padding:12px;text-align:center;font-family:'Share Tech Mono',monospace;font-size:10px;color:#7b8592;">
          INSERT .WAV OR .MP3 TO ENGAGE ARF-200 ANALYZER
        </div>
        """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# FORENSIC ENGINE EXECUTION
# ----------------------------------------------------------------------
if uploaded_file is not None and analyze_clicked:
    with st.spinner("ENGAGING ARF-200: Running Silero VAD, 50% overlapping windowing & WavLM 768-dim inference..."):
        uploaded_file.seek(0)
        file_bytes = uploaded_file.getvalue()
        data = None

        # 1. Try FastAPI backend first
        use_http = False
        if BACKEND_URL and ("127.0.0.1:8000" in BACKEND_URL or "localhost:8000" in BACKEND_URL):
            use_http = is_local_backend_active(8000)
        elif BACKEND_URL and not ("127.0.0.1" in BACKEND_URL or "localhost" in BACKEND_URL):
            use_http = True

        if use_http:
            try:
                files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                res = requests.post(f"{BACKEND_URL}/analyze", files=files, timeout=120)
                if res.status_code == 200:
                    data = res.json()
                else:
                    data = {"error": f"Backend API returned status {res.status_code}: {res.text}"}
            except Exception:
                data = None

        # 2. Standalone fallback (Streamlit Cloud zero-delay mode)
        if not data or "error" in data:
            try:
                import sys
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from backend.main import analyze_audio_data
                data = analyze_audio_data(file_bytes, uploaded_file.name)
            except Exception as e:
                import traceback
                st.error(f"Forensic Analysis Error: {e}")
                st.code(traceback.format_exc())
                st.stop()

    if not data:
        st.error("No analysis result returned. Please verify audio format (.wav / .mp3).")
        st.stop()

    if "error" in data:
        st.error(data["error"])
        st.stop()

    st.session_state["last_result"] = data
    st.session_state["analyzed_file_name"] = uploaded_file.name

    # Background DB save
    try:
        import sys
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from backend.db import save_result
        save_result(uploaded_file.name, data)
    except Exception:
        pass

    # Save to isolated session scans
    if "session_scans" not in st.session_state:
        st.session_state["session_scans"] = []
    from datetime import datetime, timezone
    created_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    st.session_state["session_scans"].insert(0, {
        "filename": uploaded_file.name,
        "duration_sec": data.get("total_duration", 0.0),
        "fake_ratio": data.get("fake_ratio", 0.0),
        "verdict": data.get("verdict", ""),
        "created_at": created_ts,
        "result": data
    })
    st.rerun()


# ----------------------------------------------------------------------
# GREEN PHOSPHOR CRT OSCILLOSCOPE SCREEN (TIMELINE CURVE)
# ----------------------------------------------------------------------
st.markdown('<div class="hardware-section-label">MAIN CRT OSCILLOSCOPE DISPLAY // CONTINUOUS PROBABILITY TRACE</div>', unsafe_allow_html=True)

if "last_result" in st.session_state:
    data = st.session_state["last_result"]
    segments = data["segments"]
    total_dur = float(data.get("total_duration", 0.0))
    fake_ratio = float(data.get("fake_ratio", 0.0))
    verdict = data.get("verdict", "")
    highest = data.get("highest_risk_segment", {})
    avg_conf = round(sum(s["confidence"] for s in segments) / max(len(segments), 1), 1)

    time_points = [round((s["start"] + s["end"]) / 2, 2) for s in segments]
    fake_probs = [round(s["fake_probability"] * 100, 2) for s in segments]
    hover_texts = [
        f"<b>WINDOW #{s['segment']+1}</b><br>"
        f"TIME: {s['start']}s – {s['end']}s<br>"
        f"AI PROBABILITY: {s['fake_probability']*100:.1f}%<br>"
        f"STATUS: <b>{s['label']}</b> ({s['confidence']}%)"
        for s in segments
    ]

    fig = go.Figure()

    # 50% Threshold Danger Area (Red phosphor wash)
    fig.add_shape(
        type="rect",
        x0=0, x1=total_dur,
        y0=50, y1=100,
        fillcolor="rgba(255, 56, 56, 0.08)",
        line=dict(width=0),
        layer="below"
    )

    # 50% Decision Line (Amber phosphor reticle)
    fig.add_trace(go.Scatter(
        x=[0, total_dur],
        y=[50, 50],
        mode="lines",
        line=dict(color="rgba(255, 180, 84, 0.8)", width=1.8, dash="dot"),
        name="AI RISK THRESHOLD (50%)",
        hoverinfo="skip"
    ))

    # Continuous Waveform Spline (Green Phosphor Glow)
    fig.add_trace(go.Scatter(
        x=time_points,
        y=fake_probs,
        mode="lines+markers",
        name="SYNTHETIC PROBABILITY",
        line=dict(color="#39ff14", width=3.5, shape="spline"),
        marker=dict(
            size=8,
            color=["#ff3838" if p >= 50 else "#39ff14" for p in fake_probs],
            line=dict(color="#031206", width=2)
        ),
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#031206",
        height=320,
        margin=dict(l=40, r=20, t=15, b=40),
        xaxis=dict(
            title=dict(text="SWEEP TIMELINE (SECONDS)", font=dict(family="Share Tech Mono", size=11, color="#39ff14")),
            tickfont=dict(family="Share Tech Mono", size=10, color="rgba(57,255,20,0.7)"),
            gridcolor="#0e3016",
            zeroline=False,
            range=[0, total_dur]
        ),
        yaxis=dict(
            title=dict(text="AI THREAT LEVEL (%)", font=dict(family="Share Tech Mono", size=11, color="#39ff14")),
            tickfont=dict(family="Share Tech Mono", size=10, color="rgba(57,255,20,0.7)"),
            gridcolor="#0e3016",
            range=[-2, 105],
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Share Tech Mono", size=10, color="#ffb454")
        ),
        hovermode="closest"
    )

    st.markdown(f"""
    <div class="crt-bezel">
      <div class="crt-screen-inner">
        <div class="crt-header">
          <span>AUDIO ARTIFACT: DEEPFAKE LOCALIZER // ARF-200</span>
          <span>FILE: {data.get('filename','CARTRIDGE')}</span>
        </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
        <div class="crt-footer">
          <span>TIME: 0:00 – {total_dur:.1f}s · WINDOWS: {len(segments)}</span>
          <span>SWEEP: 50% OVERLAP (1.0s STRIDE)</span>
          <span style="font-weight:700;color:{'#ff3838' if fake_ratio > 20 else '#39ff14'};">VERDICT: {verdict.upper()}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # VFD DIGITAL FLUORESCENT READOUT CARDS (FORENSIC SUMMARY)
    # ------------------------------------------------------------------
    st.markdown('<div class="hardware-section-label">TELEMETRY DIGITAL READOUT PANELS</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v_col = "#39ff14" if "Authentic" in verdict or "Human" in verdict else "#ff3838" if "Synthetic" in verdict or "Fake" in verdict else "#ffb454"
        st.markdown(f"""
        <div class="readout-card">
          <div class="readout-k">OVERALL VERDICT</div>
          <div class="readout-v" style="color:{v_col};font-size:18px;">{verdict}</div>
          <div class="readout-sub">Confidence: {avg_conf}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="readout-card">
          <div class="readout-k">SYNTHETIC AUDIO RATIO</div>
          <div class="readout-v" style="color:{'#ff3838' if fake_ratio > 40 else '#39ff14'};">{fake_ratio}%</div>
          <div class="readout-sub">{data.get("fake_seconds", 0)}s of {total_dur}s flagged</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        peak_prob = round(highest.get("fake_probability", 0.0) * 100, 1)
        st.markdown(f"""
        <div class="readout-card">
          <div class="readout-k">PEAK RISK SEGMENT</div>
          <div class="readout-v" style="color:{'#ff3838' if peak_prob >= 50 else '#39ff14'};">{peak_prob}% AI</div>
          <div class="readout-sub">{highest.get('start', 0)}s – {highest.get('end', 0)}s window</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="readout-card">
          <div class="readout-k">SYSTEM CONFIDENCE</div>
          <div class="readout-v" style="color:#5eead4;">{avg_conf}%</div>
          <div class="readout-sub">{len(segments)} overlapping channels</div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # MODULAR CHANNEL STRIP BREAKDOWN
    # ------------------------------------------------------------------
    st.markdown('<div class="hardware-section-label">MODULAR CHANNEL STRIP INSPECTOR</div>', unsafe_allow_html=True)

    with st.expander(f"Inspect all {len(segments)} rack channels (2.0s window / 1.0s stride)", expanded=False):
        seg_cols = st.columns(4)
        for idx, s in enumerate(segments):
            with seg_cols[idx % 4]:
                is_fake = s["label_code"] == 1
                color = "#ff3838" if is_fake else "#39ff14"
                st.markdown(f"""
                <div style="background:#121519;border:1px solid #282f37;border-left:3px solid {color};padding:8px 10px;border-radius:6px;margin-bottom:8px;">
                  <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#7b8694;">
                    CH #{s['segment']+1} · {s['start']}s–{s['end']}s
                  </div>
                  <div style="font-weight:700;font-size:13px;color:{color};margin-top:2px;">
                    {s['label']} ({s['fake_probability']*100:.1f}%)
                  </div>
                  <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#5a636f;">Confidence: {s['confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3D SONOGRAPH VOICE SIGNATURE SPECTROGRAM
    # ------------------------------------------------------------------
    st.markdown('<div class="hardware-section-label">3D SONOGRAPH VOICE SIGNATURE TOPOGRAPHY</div>', unsafe_allow_html=True)

    spec_data = data.get("spectrogram", {})
    if "z" in spec_data and spec_data["z"]:
        z = np.array(spec_data["z"])
        x = np.array(spec_data.get("time", list(range(z.shape[1]))))
        y = np.array(spec_data.get("mel", list(range(z.shape[0]))))

        fig3d = go.Figure(data=[go.Surface(
            z=z, x=x, y=y,
            colorscale=[
                [0, "#031206"],
                [0.35, "#0b3815"],
                [0.7, "#1d7e0c"],
                [1.0, "#39ff14"]
            ],
            showscale=False
        )])
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title="Time (s)", showbackground=False, color="#39ff14"),
                yaxis=dict(title="Mel Bands", showbackground=False, color="#39ff14"),
                zaxis=dict(title="dB", showbackground=False, color="#39ff14"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=360,
        )
        st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": False})

else:
    # Standby Display before analysis
    st.markdown("""
    <div class="crt-bezel">
      <div class="crt-screen-inner" style="text-align:center;padding:50px 20px;">
        <div class="crt-header" style="justify-content:center;">
          AUDIO ARTIFACT: DEEPFAKE LOCALIZER // ARF-200
        </div>
        <div style="font-family:'VT323',monospace;font-size:32px;color:#39ff14;text-shadow:0 0 10px rgba(57,255,20,0.8);margin:24px 0 12px;">
          SYSTEM READY // AWAITING AUDIO INPUT
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:rgba(57,255,20,0.65);letter-spacing:0.12em;">
          INSERT .WAV OR .MP3 CARTRIDGE INTO MEDIA BAY ABOVE AND PRESS [◈ ANALYZE CARTRIDGE]
        </div>
        <div class="crt-footer" style="margin-top:30px;">
          <span>VAD: STANDBY</span>
          <span>WAVLM 768-DIM FOUNDATION MODEL: LOADED</span>
          <span>READY</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# Close Outer Rack Frame
st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# ACTIVE SESSION SCAN HISTORY (ISOLATED TO CURRENT BROWSER WINDOW)
# ----------------------------------------------------------------------
st.markdown('<div class="hardware-section-label">RACK DATALOGGER // ACTIVE SESSION HISTORY</div>', unsafe_allow_html=True)

session_scans = st.session_state.get("session_scans", [])

col_hist_info, col_hist_act = st.columns([3, 1])
with col_hist_info:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <span style="font-family:'Share Tech Mono',monospace;font-size:11px;background:#14171b;border:1px solid #2d333b;color:#5eead4;padding:4px 10px;border-radius:4px;">
        LOG UNIT #{st.session_state['session_token']}
      </span>
      <span style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#7a8694;">
        Ephemeral telemetry: records are private to this window and auto-wipe on tab close.
      </span>
    </div>
    """, unsafe_allow_html=True)

with col_hist_act:
    if session_scans:
        if st.button("🗑️ PURGE LOGS", use_container_width=True):
            st.session_state["session_scans"] = []
            if "last_result" in st.session_state:
                del st.session_state["last_result"]
            st.rerun()

if session_scans:
    with st.expander(f"Datalogger Records ({len(session_scans)} tape{'s' if len(session_scans) > 1 else ''})", expanded=True):
        for idx, h in enumerate(session_scans):
            v = h.get("verdict", "")
            col_v = "#39ff14" if "Authentic" in v or "Human" in v else "#ff3838" if "Synthetic" in v or "Fake" in v else "#ffb454"
            c_info, c_btn = st.columns([4, 1])
            with c_info:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;background:#14171b;border:1px solid #292f37;border-radius:6px;padding:10px 14px;margin-bottom:6px;">
                  <div>
                    <div style="font-weight:700;font-size:13px;color:#e3e8ef;">{h.get('filename')}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:10.5px;color:#7a8694;margin-top:2px;">
                      Duration: {h.get('duration_sec', 0):.1f}s · Logged at {str(h.get('created_at', ''))[11:19]} UTC
                    </div>
                  </div>
                  <div style="text-align:right;">
                    <span style="font-family:'Share Tech Mono',monospace;font-size:11px;font-weight:700;color:{col_v};background:#1c2026;border:1px solid #2d333b;padding:3px 8px;border-radius:4px;">
                      {v or 'LOGGED'}
                    </span>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:10.5px;color:#7a8694;margin-top:2px;">
                      Fake Ratio: {h.get('fake_ratio', 0):.1f}%
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with c_btn:
                if st.button("Inspect ◈", key=f"reinspect_{idx}", use_container_width=True):
                    st.session_state["last_result"] = h.get("result", {})
                    st.session_state["analyzed_file_name"] = h.get("filename", "")
                    st.rerun()
else:
    st.markdown("""
    <div style="background:#14171b;border:1.5px dashed #292f37;border-radius:8px;padding:24px;text-align:center;color:#7a8694;font-family:'Share Tech Mono',monospace;font-size:12px;">
      NO RECORD LOGS IN THIS SESSION. INSERT CARTRIDGE AND ANALYZE TO POPULATE DATALOGGER.
    </div>
    """, unsafe_allow_html=True)
