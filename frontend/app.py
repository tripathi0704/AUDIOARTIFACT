"""
app.py — AudioArtifact Scanner Interface
Upload audio -> send to backend -> render a dark, instrument-style
scanner report (verdict pill, confidence gauge, animated timeline,
3D voice-signature graph).
"""

import io
import json

import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

BACKEND_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="AudioArtifact", page_icon="◈", layout="wide")

# ----------------------------------------------------------------------
# GLOBAL STYLE — dark instrument theme + hide default Streamlit chrome
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#07090A; --panel:#121614; --panel-2:#181D1A; --line:#242B25;
  --text:#ECEFEB; --muted:#8B958D; --muted-dim:#565F58;
  --real:#33D17E; --real-dim: rgba(51,209,126,0.13);
  --fake:#FF5C5C; --fake-dim: rgba(255,92,92,0.13);
  --amber:#FFB454; --amber-dim: rgba(255,180,84,0.14);
  --scan:#5EEAD4;
}

#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility:hidden; height:0;}
.stApp{ background:var(--bg); }
.block-container{ padding-top:2.4rem; max-width:1080px; }
html, body, [class*="css"]{ font-family:'Manrope', sans-serif; color:var(--text); }

.brand-row{ display:flex; align-items:center; justify-content:space-between; margin-bottom:28px; }
.brand{ font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:19px; display:flex; align-items:center; gap:10px; }
.brand .px{ width:9px; height:9px; background:var(--real); border-radius:50%; box-shadow:0 0 10px var(--real); animation:pulse 2.2s infinite; }
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.3;}}
.status-tag{ font-family:'IBM Plex Mono', monospace; font-size:11px; color:var(--muted-dim); display:flex; align-items:center; gap:7px; }
.status-tag i{ width:6px; height:6px; border-radius:50%; background:var(--real); box-shadow:0 0 6px var(--real); }

h1.hero-title{
  font-family:'Space Grotesk', sans-serif; font-weight:700; letter-spacing:-0.02em;
  font-size:clamp(28px,4vw,42px); line-height:1.15; margin-bottom:6px;
}
p.hero-sub{ color:var(--muted); font-size:15px; max-width:560px; margin-bottom:30px; }

/* file uploader restyle */
[data-testid="stFileUploader"]{
  background:var(--panel); border:1.5px dashed var(--line); border-radius:14px;
  padding:6px 10px;
}
[data-testid="stFileUploader"] section{ background:transparent; border:none; }
[data-testid="stFileUploaderDropzoneInstructions"] svg{ display:none; }

/* audio player container */
[data-testid="stAudio"]{ margin-top:14px; }

/* analyze button */
div.stButton > button{
  background:var(--text); color:#0A0C0B; border:none; font-weight:700;
  font-family:'Space Grotesk', sans-serif; font-size:14px; padding:11px 26px;
  border-radius:22px; transition: transform 0.15s ease;
}
div.stButton > button:hover{ transform:translateY(-1px); }

.section-label{
  font-family:'IBM Plex Mono', monospace; font-size:11.5px; letter-spacing:0.16em; text-transform:uppercase;
  color:var(--scan); margin:36px 0 12px; display:flex; align-items:center; gap:10px;
}
.section-label::before{ content:''; width:22px; height:1px; background:var(--scan); }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="brand-row">
  <div class="brand"><span class="px"></span>AudioArtifact</div>
  <div class="status-tag"><i></i>SCANNER ONLINE</div>
</div>
<h1 class="hero-title">Every voice leaves a signature.</h1>
<p class="hero-sub">Drop a clip below to scan it segment by segment and see exactly
where — if anywhere — it turns synthetic.</p>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# UPLOAD
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader(" ", type=["mp3", "wav"], label_visibility="collapsed")

if uploaded_file is not None:
    st.audio(uploaded_file)
    analyze_clicked = st.button("Analyze", type="primary")
else:
    analyze_clicked = False

# ----------------------------------------------------------------------
# ANALYSIS
# ----------------------------------------------------------------------
if uploaded_file is not None and analyze_clicked:
    with st.spinner(" "):
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        try:
            response = requests.post(BACKEND_URL, files=files, timeout=120)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            st.error(f"Could not reach the scanner service: {e}")
            st.stop()

    if "error" in result:
        st.error(result["error"])
        st.stop()

    segments = result["segments"]
    total_duration = result["total_duration"]
    fake_seconds = result["fake_seconds"]

    if fake_seconds == 0:
        verdict_class, verdict_label = "is-real", "Human-verified"
    elif fake_seconds >= total_duration:
        verdict_class, verdict_label = "is-fake", "Fully synthetic"
    else:
        verdict_class, verdict_label = "is-mixed", "Mixed signal detected"

    avg_conf = round(sum(s["confidence"] for s in segments) / max(len(segments), 1), 1)

    # ------------------------------------------------------------
    # SCANNER COMPONENT (self-contained HTML/CSS/JS, real data)
    # ------------------------------------------------------------
    segments_json = json.dumps([
        {"start": s["start"], "end": s["end"],
         "verdict": "real" if s["label_code"] == 0 else "fake",
         "confidence": s["confidence"]}
        for s in segments
    ])

    scanner_html = """
    <div id="root"></div>
    <style>
      body{ margin:0; font-family:'Manrope', sans-serif; background:transparent; }
      .mono{ font-family:'IBM Plex Mono', monospace; }
      .shell{ background:linear-gradient(180deg, #121614, #0B0E0C); border:1px solid #242B25; border-radius:18px; overflow:hidden; color:#ECEFEB; }
      .topbar{ display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap; padding:16px 20px; border-bottom:1px solid #242B25; }
      .fname{ font-size:13.5px; font-weight:600; }
      .fsub{ font-family:'IBM Plex Mono', monospace; font-size:11px; color:#565F58; margin-top:2px; }
      .pill{ font-family:'IBM Plex Mono', monospace; font-size:11.5px; font-weight:600; padding:7px 14px; border-radius:20px; display:flex; align-items:center; gap:8px; }
      .pill i{ width:7px; height:7px; border-radius:50%; }
      .pill.is-real{ background:rgba(51,209,126,0.13); border:1px solid rgba(51,209,126,0.35); color:#33D17E; }
      .pill.is-real i{ background:#33D17E; box-shadow:0 0 8px #33D17E; }
      .pill.is-fake{ background:rgba(255,92,92,0.13); border:1px solid rgba(255,92,92,0.35); color:#FF5C5C; }
      .pill.is-fake i{ background:#FF5C5C; box-shadow:0 0 8px #FF5C5C; }
      .pill.is-mixed{ background:rgba(255,180,84,0.14); border:1px solid rgba(255,180,84,0.35); color:#FFB454; }
      .pill.is-mixed i{ background:#FFB454; box-shadow:0 0 8px #FFB454; }
      .body{ padding:24px 20px 6px; }
      .canvas{ position:relative; height:150px; display:flex; align-items:center; overflow:hidden; border-radius:12px;
        background:repeating-linear-gradient(180deg, transparent, transparent 38px, rgba(236,239,235,0.06) 39px); }
      .baseline{ position:absolute; left:0; right:0; top:50%; height:1px; background:#242B25; }
      .track{ position:relative; z-index:2; display:flex; align-items:center; gap:2px; width:100%; height:100%; padding:0 2px; }
      .bar{ flex:1; min-width:2px; border-radius:2px; background:#565F58; opacity:0.35; transform-origin:center; transition:background 0.35s ease, opacity 0.35s ease, transform 0.5s ease; }
      .scanline{ position:absolute; top:0; bottom:0; width:2px; left:0%; background:#5EEAD4; box-shadow:0 0 18px 3px #5EEAD4; z-index:3; opacity:0; transition:left 1.6s cubic-bezier(.3,.6,.2,1); }
      .scanline.on{ opacity:1; }
      .ruler{ display:flex; justify-content:space-between; margin-top:9px; font-family:'IBM Plex Mono', monospace; font-size:10.5px; color:#565F58; }
      .bottom{ display:flex; flex-wrap:wrap; gap:20px; align-items:center; justify-content:space-between; padding:18px 20px; border-top:1px solid #242B25; }
      .legend{ display:flex; gap:16px; flex-wrap:wrap; font-size:12.5px; color:#8B958D; }
      .legend span{ display:inline-flex; align-items:center; gap:7px; }
      .legend i{ width:8px;height:8px;border-radius:50%; display:inline-block; }
      .gauge-wrap{ display:flex; align-items:center; gap:12px; }
      .gauge-wrap svg{ width:52px; height:52px; transform:rotate(-90deg); }
      .gauge-bg{ fill:none; stroke:#242B25; stroke-width:6; }
      .gauge-fg{ fill:none; stroke:#33D17E; stroke-width:6; stroke-linecap:round; stroke-dasharray:251.2; stroke-dashoffset:251.2; transition:stroke-dashoffset 1.1s cubic-bezier(.3,.6,.2,1); }
      .gauge-num{ font-size:17px; font-weight:700; font-family:'Space Grotesk', sans-serif; }
      .gauge-cap{ font-family:'IBM Plex Mono', monospace; font-size:10px; color:#565F58; text-transform:uppercase; }
      .tooltip{ position:absolute; z-index:10; transform:translate(-50%,-108%); background:#080A09; border:1px solid #242B25; border-radius:8px; padding:8px 11px; font-family:'IBM Plex Mono', monospace; font-size:11px; white-space:nowrap; pointer-events:none; opacity:0; transition:opacity 0.15s ease; }
      .tooltip.show{ opacity:1; }
    </style>

    <div class="shell">
      <div class="topbar">
        <div><div class="fname">__FILENAME__</div><div class="fsub mono">__DURATION__s scanned</div></div>
        <div class="pill __VERDICT_CLASS__"><i></i><span>__VERDICT_LABEL__</span></div>
      </div>
      <div class="body">
        <div class="canvas" id="canvas">
          <div class="baseline"></div>
          <div class="track" id="track"></div>
          <div class="scanline" id="scanline"></div>
          <div class="tooltip" id="tooltip"></div>
        </div>
        <div class="ruler" id="ruler"></div>
      </div>
      <div class="bottom">
        <div class="legend">
          <span><i style="background:#33D17E"></i>Human-verified</span>
          <span><i style="background:#FF5C5C"></i>AI-synthesized</span>
        </div>
        <div class="gauge-wrap">
          <svg viewBox="0 0 90 90"><circle class="gauge-bg" cx="45" cy="45" r="40"/><circle class="gauge-fg" id="gaugeFg" cx="45" cy="45" r="40"/></svg>
          <div><div class="gauge-num" id="gaugeNum">0%</div><div class="gauge-cap">confidence</div></div>
        </div>
      </div>
    </div>

    <script>
      const segments = __SEGMENTS_JSON__;
      const track = document.getElementById('track');
      const ruler = document.getElementById('ruler');
      const tooltip = document.getElementById('tooltip');
      const scanline = document.getElementById('scanline');
      const CIRC = 251.2;

      segments.forEach((seg, idx) => {
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.dataset.idx = idx;
        bar.style.height = (28 + (seg.confidence % 40)) + 'px';
        track.appendChild(bar);
      });

      const totalDur = segments.length ? segments[segments.length-1].end : 0;
      const steps = 6;
      for(let i=0;i<=steps;i++){
        const s = document.createElement('span');
        const t = (totalDur/steps)*i;
        const m = Math.floor(t/60), sec = Math.floor(t%60);
        s.textContent = m + ':' + String(sec).padStart(2,'0');
        ruler.appendChild(s);
      }

      track.addEventListener('mousemove', e => {
        const bar = e.target.closest('.bar');
        if(!bar){ tooltip.classList.remove('show'); return; }
        const seg = segments[bar.dataset.idx];
        const rect = track.getBoundingClientRect();
        tooltip.style.left = (e.clientX - rect.left) + 'px';
        tooltip.style.top = '0px';
        tooltip.innerHTML = (seg.verdict==='real' ? 'Human-verified' : 'AI-synthesized') +
          '<div style="color:#8B958D;margin-top:2px;">' + seg.start.toFixed(1) + 's\u2013' + seg.end.toFixed(1) + 's \u00b7 ' + seg.confidence + '%</div>';
        tooltip.classList.add('show');
      });
      track.addEventListener('mouseleave', () => tooltip.classList.remove('show'));

      window.addEventListener('load', () => {
        setTimeout(() => {
          scanline.classList.add('on');
          scanline.style.left = '100%';
          const bars = Array.from(track.children);
          bars.forEach((bar, idx) => {
            const delay = (idx/bars.length) * 1500;
            setTimeout(() => {
              const seg = segments[idx];
              bar.style.background = seg.verdict === 'real' ? '#33D17E' : '#FF5C5C';
              bar.style.opacity = '0.92';
            }, delay);
          });
          setTimeout(() => {
            scanline.classList.remove('on');
            const gaugeFg = document.getElementById('gaugeFg');
            const gaugeNum = document.getElementById('gaugeNum');
            const val = __AVG_CONF__;
            const offset = CIRC * (1 - val/100);
            gaugeFg.style.strokeDashoffset = offset;
            gaugeFg.style.stroke = val >= 70 ? '#33D17E' : '#FFB454';
            gaugeNum.textContent = val + '%';
          }, 1600);
        }, 250);
      });
    </script>
    """

    scanner_html = (scanner_html
        .replace("__FILENAME__", result["filename"])
        .replace("__DURATION__", str(round(total_duration)))
        .replace("__VERDICT_CLASS__", verdict_class)
        .replace("__VERDICT_LABEL__", verdict_label)
        .replace("__SEGMENTS_JSON__", segments_json)
        .replace("__AVG_CONF__", str(avg_conf))
    )

    components.html(scanner_html, height=380, scrolling=False)

    # ------------------------------------------------------------
    # 3D VOICE SIGNATURE GRAPH
    # ------------------------------------------------------------
    st.markdown('<div class="section-label">Voice Signature</div>', unsafe_allow_html=True)

    uploaded_file.seek(0)
    y, sr = librosa.load(io.BytesIO(uploaded_file.read()), sr=16000, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    S_db = librosa.power_to_db(S, ref=np.max)

    # downsample time axis for a smooth, light-weight 3D render
    max_frames = 120
    if S_db.shape[1] > max_frames:
        idx = np.linspace(0, S_db.shape[1] - 1, max_frames).astype(int)
        S_db = S_db[:, idx]

    time_axis = np.linspace(0, total_duration, S_db.shape[1])
    mel_axis = np.arange(S_db.shape[0])

    fig = go.Figure(data=[go.Surface(
        z=S_db, x=time_axis, y=mel_axis,
        colorscale=[[0, "#07090A"], [0.4, "#123024"], [0.7, "#1e6b4c"], [1, "#33D17E"]],
        showscale=False,
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="", showbackground=False, color="#565F58"),
            yaxis=dict(title="", showbackground=False, color="#565F58"),
            zaxis=dict(title="", showbackground=False, color="#565F58"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
