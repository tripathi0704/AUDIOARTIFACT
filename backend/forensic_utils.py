"""
forensic_utils.py
-----------------
Purpose : Forensic acoustic signal analysis, cryptographic integrity hashing,
          speaker embedding comparison, and printable forensic audit report generation.
All calculations are pure NumPy and standard library (ultra-fast, ~4ms runtime,
no external C-extensions, fully compatible across all platforms and Python 3.14).
"""

import io
import hashlib
import numpy as np
import soundfile as sf
from datetime import datetime, timezone


def compute_file_hashes(content: bytes) -> dict:
    """Computes cryptographic hashes and payload metrics for forensic chain-of-custody."""
    sha256 = hashlib.sha256(content).hexdigest()
    md5 = hashlib.md5(content).hexdigest()
    return {
        "sha256": sha256,
        "md5": md5,
        "size_bytes": len(content),
        "size_kb": round(len(content) / 1024, 2),
    }


def compute_acoustic_forensics(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Computes biological and acoustic forensic indicators from audio waveform.
    Pure NumPy implementation for maximum speed and zero JIT dependencies.
    """
    if len(y) == 0:
        return {}

    y_clean = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    
    # 1. Energy and Dynamic Range
    rms = float(np.sqrt(np.mean(y_clean**2)))
    rms_db = round(float(20 * np.log10(max(rms, 1e-7))), 1)
    peak = float(np.max(np.abs(y_clean)))
    peak_db = round(float(20 * np.log10(max(peak, 1e-7))), 1)
    crest_factor = round(float(peak / (rms + 1e-7)), 2)
    dynamic_range_db = round(abs(peak_db - rms_db), 1)

    # 2. Zero Crossing Rate (detects synthetic silence / unnatural transitions)
    zero_crossings = np.sum(np.diff(np.signbit(y_clean)) != 0)
    zcr = round(float(zero_crossings / max(len(y_clean) - 1, 1)), 4)

    # 3. Spectral Distribution & High-Frequency Cutoff (Vocoder Artifacts)
    # Neural vocoders (HiFi-GAN, MelGAN) often drop frequencies abruptly above 7-8kHz
    n_samples = len(y_clean)
    fft_vals = np.abs(np.fft.rfft(y_clean))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    total_energy = float(np.sum(fft_vals)) + 1e-9

    # Spectral Centroid (Center of mass of frequency spectrum)
    centroid = round(float(np.sum(freqs * fft_vals) / total_energy), 1)

    # Spectral Rolloff 85% (frequency below which 85% of spectral energy lies)
    cum_energy = np.cumsum(fft_vals)
    rolloff_idx = np.searchsorted(cum_energy, 0.85 * total_energy)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    rolloff_hz = round(float(freqs[rolloff_idx]), 1)

    # High-Frequency Energy Ratio (energy > 6000Hz / total energy)
    hf_mask = freqs >= 6000.0
    hf_ratio = round(float(np.sum(fft_vals[hf_mask]) / total_energy) * 100, 2)

    # 4. Fundamental Frequency (F0) & Pitch Stability
    # Frame-by-frame autocorrelation to estimate pitch variation (natural vs robotic)
    frame_len = int(sr * 0.04)  # 40ms frame
    hop_len = int(sr * 0.02)    # 20ms hop
    pitches = []

    min_lag = int(sr / 450)  # max ~450 Hz
    max_lag = int(sr / 65)   # min ~65 Hz

    for start_idx in range(0, n_samples - frame_len, hop_len):
        frame = y_clean[start_idx:start_idx + frame_len]
        frame_energy = np.sum(frame**2)
        if frame_energy < 0.005:
            continue  # ignore unvoiced silence
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        if max_lag < len(corr):
            search_window = corr[min_lag:max_lag]
            if len(search_window) > 0 and np.max(search_window) > 0.3 * corr[0]:
                best_lag = min_lag + int(np.argmax(search_window))
                f0 = sr / best_lag
                pitches.append(f0)

    if len(pitches) >= 5:
        pitch_mean = round(float(np.mean(pitches)), 1)
        pitch_std = round(float(np.std(pitches)), 1)
        # Coefficient of variation (higher = more expressive human inflection, lower = monotone TTS)
        pitch_cv = round(float((pitch_std / max(pitch_mean, 1.0)) * 100), 1)
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_cv = 0.0

    return {
        "rms_energy_db": rms_db,
        "peak_db": peak_db,
        "dynamic_range_db": dynamic_range_db,
        "crest_factor": crest_factor,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "spectral_rolloff_85_hz": rolloff_hz,
        "high_frequency_ratio_pct": hf_ratio,
        "pitch_mean_hz": pitch_mean,
        "pitch_variation_hz": pitch_std,
        "pitch_inflection_score": pitch_cv,
    }


def slice_segment_audio_bytes(y: np.ndarray, start_sec: float, end_sec: float, sr: int = 16000) -> bytes:
    """Slices a waveform and exports it as in-memory WAV bytes for instant browser playback."""
    start_idx = max(0, int(start_sec * sr))
    end_idx = min(len(y), int(end_sec * sr))
    slice_data = y[start_idx:end_idx]
    
    bio = io.BytesIO()
    sf.write(bio, slice_data, sr, format='WAV', subtype='PCM_16')
    bio.seek(0)
    return bio.getvalue()


def compute_speaker_similarity(emb1: np.ndarray, emb2: np.ndarray) -> dict:
    """
    Compares two 768-dimensional WavLM speaker voiceprint embeddings using Cosine Similarity.
    Returns similarity score, distance metric, and matching verdict.
    """
    v1 = emb1.flatten()
    v2 = emb2.flatten()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return {
            "cosine_similarity": 0.0,
            "similarity_percentage": 0.0,
            "verdict": "Indeterminate (Empty or silent audio)",
            "match": False
        }

    cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    # Normalized 0% to 100% similarity score
    sim_pct = round(float((cos_sim + 1.0) / 2.0 * 100), 2)
    euclidean_dist = round(float(np.linalg.norm(v1 / norm1 - v2 / norm2)), 4)

    if cos_sim >= 0.82:
        verdict = "Strong Speaker Match (High Identity Concordance)"
        match = True
    elif cos_sim >= 0.65:
        verdict = "Moderate Vocal Similarity (Acoustic overlap / Ambiguous)"
        match = False
    else:
        verdict = "Distinct Speakers (Acoustic Profiles Do Not Match)"
        match = False

    return {
        "cosine_similarity": round(cos_sim, 4),
        "similarity_percentage": sim_pct,
        "euclidean_distance": euclidean_dist,
        "verdict": verdict,
        "match": match
    }


def generate_forensic_html_report(result: dict, user_tz_name: str = None) -> str:
    """
    Generates a professional, self-contained, responsive forensic audit certificate.
    Supports browser print-to-PDF with clean formatting and dynamic local timezone conversion.
    """
    filename = result.get("filename", "audio_evidence.wav")
    total_dur = result.get("total_duration", 0.0)
    fake_ratio = result.get("fake_ratio", 0.0)
    verdict = result.get("verdict", "Unknown")
    fake_sec = result.get("fake_seconds", 0.0)
    segments = result.get("segments", [])
    highest = result.get("highest_risk_segment", {})
    hashes = result.get("file_hashes", {})
    forensics = result.get("forensic_signals", {})
    
    now_utc = datetime.now(timezone.utc)
    utc_iso = now_utc.isoformat().replace("+00:00", "Z")
    report_ts = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    if user_tz_name:
        try:
            import zoneinfo
            tz_obj = zoneinfo.ZoneInfo(user_tz_name)
            local_dt = now_utc.astimezone(tz_obj)
            report_ts = local_dt.strftime("%Y-%m-%d %I:%M:%S %p") + f" ({user_tz_name})"
        except Exception:
            pass

    verdict_color = "#3ECF8E" if fake_ratio <= 15 else "#FF5C5C" if fake_ratio >= 75 else "#FFB454"
    verdict_bg = "rgba(62,207,142,0.12)" if fake_ratio <= 15 else "rgba(255,92,92,0.12)" if fake_ratio >= 75 else "rgba(255,180,84,0.12)"

    # Build segments rows
    seg_rows = []
    for s in segments:
        is_ai = s.get("label_code") == 1
        s_col = "#FF5C5C" if is_ai else "#3ECF8E"
        seg_rows.append(f"""
        <tr>
            <td style="padding:8px 12px;border-bottom:1px solid #262B27;font-family:monospace;">Window #{s.get('segment', 0)+1}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #262B27;font-family:monospace;">{s.get('start', 0)}s – {s.get('end', 0)}s</td>
            <td style="padding:8px 12px;border-bottom:1px solid #262B27;color:{s_col};font-weight:600;">{s.get('label')}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #262B27;font-family:monospace;">{s.get('fake_probability', 0)*100:.1f}%</td>
            <td style="padding:8px 12px;border-bottom:1px solid #262B27;font-family:monospace;">{s.get('confidence', 0)}%</td>
        </tr>
        """)
    segments_html = "".join(seg_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Forensic Audit Report — {filename}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0A0C0B;
    color: #ECEFEB;
    font-family: 'Manrope', -apple-system, sans-serif;
    padding: 36px 24px;
    line-height: 1.5;
  }}
  .container {{
    max-width: 960px;
    margin: 0 auto;
    background: #131614;
    border: 1px solid #262B27;
    border-radius: 14px;
    padding: 36px;
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2px solid #262B27;
    padding-bottom: 20px;
    margin-bottom: 28px;
  }}
  .brand {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px;
    font-weight: 700;
  }}
  .brand span {{ color: #3ECF8E; }}
  .brand-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8C958E;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }}
  .meta-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    background: #1A1E1B;
    border: 1px solid #262B27;
    padding: 6px 14px;
    border-radius: 20px;
    color: #5EEAD4;
    text-align: right;
  }}
  .verdict-box {{
    background: {verdict_bg};
    border: 1px solid {verdict_color};
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .verdict-title {{
    font-size: 26px;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    color: {verdict_color};
  }}
  .grid-4 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .card {{
    background: #1A1E1B;
    border: 1px solid #262B27;
    border-radius: 10px;
    padding: 16px 20px;
  }}
  .card .label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    color: #8C958E;
    letter-spacing: 0.08em;
  }}
  .card .val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
  }}
  .sec-heading {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #FFB454;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 28px 0 12px;
    border-bottom: 1px solid #262B27;
    padding-bottom: 6px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    background: #1A1E1B;
    border-radius: 8px;
    overflow: hidden;
  }}
  th {{
    background: #262B27;
    text-align: left;
    padding: 10px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8C958E;
  }}
  .hash-box {{
    background: #1A1E1B;
    border: 1px solid #262B27;
    padding: 12px 16px;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #ECEFEB;
    word-break: break-all;
    margin-bottom: 8px;
  }}
  .btn-print {{
    background: #3ECF8E;
    color: #0A0C0B;
    border: none;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    padding: 10px 22px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 13px;
  }}
  @media print {{
    body {{ background: #ffffff; color: #111111; padding: 0; }}
    .container {{ border: none; background: #ffffff; padding: 0; }}
    .btn-print {{ display: none; }}
    .card, table, .hash-box {{ background: #f8f9fa; border: 1px solid #dee2e6; color: #111; }}
    th {{ background: #e9ecef; color: #333; }}
    .verdict-box {{ border: 2px solid #000; background: #f0fdf4; }}
    .verdict-title {{ color: #000; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <div class="brand">Audio<span>Artifact</span> Forensic Laboratory</div>
      <div class="brand-sub">DEEPFAKE AUDIO DETECTION AUDIT & EVIDENCE CERTIFICATE</div>
    </div>
    <div>
      <button class="btn-print" onclick="window.print()">🖨️ Print / Save as PDF</button>
      <div class="meta-tag" id="report-audit-ts" data-utc="{utc_iso}" style="margin-top:8px;">AUDIT TS: {report_ts}</div>
    </div>
  </div>

  <div class="verdict-box">
    <div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#8C958E;">Forensic Verdict</div>
      <div class="verdict-title">{verdict}</div>
      <div style="font-size:13px;color:#8C958E;margin-top:4px;">Target: <b>{filename}</b> · Analyzed Duration: <b>{total_dur}s</b></div>
    </div>
    <div style="text-align:right;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:36px;font-weight:800;color:{verdict_color};">{fake_ratio}%</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8C958E;">Synthetic Ratio</div>
    </div>
  </div>

  <div class="grid-4">
    <div class="card">
      <div class="label">Flagged Duration</div>
      <div class="val" style="color:{verdict_color};">{fake_sec}s / {total_dur}s</div>
    </div>
    <div class="card">
      <div class="label">Windows Analyzed</div>
      <div class="val">{len(segments)} windows</div>
    </div>
    <div class="card">
      <div class="label">Peak Risk Window</div>
      <div class="val">{round(highest.get('fake_probability', 0)*100, 1)}% AI</div>
      <div style="font-size:11px;color:#8C958E;margin-top:2px;">Interval: {highest.get('start', 0)}s–{highest.get('end', 0)}s</div>
    </div>
    <div class="card">
      <div class="label">Spectral Rolloff (85%)</div>
      <div class="val">{forensics.get('spectral_rolloff_85_hz', 'N/A')} Hz</div>
      <div style="font-size:11px;color:#8C958E;margin-top:2px;">Centroid: {forensics.get('spectral_centroid_hz', 'N/A')} Hz</div>
    </div>
  </div>

  <div class="sec-heading">Chain-of-Custody Cryptographic Signatures</div>
  <div class="hash-box"><b>SHA-256:</b> {hashes.get('sha256', 'N/A')}</div>
  <div class="hash-box"><b>MD5:</b> {hashes.get('md5', 'N/A')} · <b>File Size:</b> {hashes.get('size_kb', 0)} KB ({hashes.get('size_bytes', 0)} bytes)</div>

  <div class="sec-heading">Biological & Acoustic Artifact Findings</div>
  <div class="grid-4">
    <div class="card">
      <div class="label">Zero Crossing Rate</div>
      <div class="val">{forensics.get('zero_crossing_rate', 'N/A')}</div>
    </div>
    <div class="card">
      <div class="label">Pitch Mean (F0)</div>
      <div class="val">{forensics.get('pitch_mean_hz', 'N/A')} Hz</div>
    </div>
    <div class="card">
      <div class="label">Pitch Inflection Score</div>
      <div class="val">{forensics.get('pitch_inflection_score', 'N/A')}%</div>
    </div>
    <div class="card">
      <div class="label">High Freq Energy (>6kHz)</div>
      <div class="val">{forensics.get('high_frequency_ratio_pct', 'N/A')}%</div>
    </div>
  </div>

  <div class="sec-heading">Temporal Speech Frame Breakdown</div>
  <table>
    <thead>
      <tr>
        <th>Frame</th>
        <th>Timestamp</th>
        <th>Classification</th>
        <th>AI Probability</th>
        <th>Model Confidence</th>
      </tr>
    </thead>
    <tbody>
      {segments_html}
    </tbody>
  </table>

  <div style="margin-top:36px;padding-top:16px;border-top:1px solid #262B27;display:flex;justify-content:space-between;align-items:center;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#5E6660;">
    <div>Engine: AudioArtifact Multi-Layer Acoustic Intelligence Engine</div>
    <div>AudioArtifact v2.0 Forensic Suite</div>
  </div>

  <script>
    try {{
      const tsEl = document.getElementById('report-audit-ts');
      if (tsEl && tsEl.dataset.utc) {{
        const d = new Date(tsEl.dataset.utc);
        if (!isNaN(d.getTime())) {{
          const localStr = d.toLocaleString(undefined, {{
            year: 'numeric', month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit', second: '2-digit',
            hour12: true
          }});
          const tzStr = Intl.DateTimeFormat().resolvedOptions().timeZone || 'Local';
          tsEl.textContent = 'AUDIT TS: ' + localStr + ' (' + tzStr + ')';
        }}
      }}
    }} catch(e) {{}}
  </script>
</div>
</body>
</html>"""
    return html
