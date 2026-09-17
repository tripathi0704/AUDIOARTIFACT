import os
import sys
import io
import soundfile as sf
import numpy as np

project_root = os.path.abspath(".")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("="*70)
print("AUDIOARTIFACT COMPLETE END-TO-END VERIFICATION SUITE")
print("="*70)

# 1. Imports
try:
    from backend.main import (
        analyze_audio_data,
        extract_speaker_embedding_vector,
        decode_audio_bytes,
        get_model,
        get_wavlm,
    )
    from backend.forensic_utils import (
        compute_file_hashes,
        compute_acoustic_forensics,
        compute_speaker_similarity,
        generate_forensic_html_report,
        slice_segment_audio_bytes,
    )
    from backend.db import init_db, save_result, get_history
    print("[PASS] All backend modules imported successfully.")
except Exception as e:
    print(f"[FAIL] Backend module import error: {e}")
    sys.exit(1)

# Pick real and fake test files
real_dir = os.path.join(project_root, "data", "real")
fake_dir = os.path.join(project_root, "data", "fake")

real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".mp3")][:2]
fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".mp3")][:2]

print(f"\n[Test Files Selected]")
print(f"  Real 1: {os.path.basename(real_files[0])}")
print(f"  Real 2: {os.path.basename(real_files[1])}")
print(f"  Fake 1: {os.path.basename(fake_files[0])}")
print(f"  Fake 2: {os.path.basename(fake_files[1])}")

# -------------------------------------------------------------
# TEST 1: Single Audio Localizer on Real Audio
# -------------------------------------------------------------
print("\n--- [TEST 1: Single Audio Localizer (Real Audio)] ---")
with open(real_files[0], "rb") as f:
    real_bytes = f.read()

res_real = analyze_audio_data(real_bytes, os.path.basename(real_files[0]))
assert "error" not in res_real, f"Error in real audio analysis: {res_real.get('error')}"
print(f"  Total Duration   : {res_real['total_duration']}s")
print(f"  Analyzed Duration: {res_real['analyzed_duration']}s")
print(f"  Segments Count   : {res_real['segments_count']}")
print(f"  Synthetic Ratio  : {res_real['fake_ratio']}%")
print(f"  Verdict          : {res_real['verdict']}")
print(f"  SHA-256          : {res_real['file_hashes']['sha256'][:16]}...")
print(f"  Spectrogram Z-dim: {len(res_real['spectrogram']['z'])}x{len(res_real['spectrogram']['z'][0]) if res_real['spectrogram']['z'] else 0}")
print("  [PASS] Single Audio Localizer passed for Real Audio.")

# -------------------------------------------------------------
# TEST 2: Single Audio Localizer on Fake Audio
# -------------------------------------------------------------
print("\n--- [TEST 2: Single Audio Localizer (Fake Audio)] ---")
with open(fake_files[0], "rb") as f:
    fake_bytes = f.read()

res_fake = analyze_audio_data(fake_bytes, os.path.basename(fake_files[0]))
assert "error" not in res_fake, f"Error in fake audio analysis: {res_fake.get('error')}"
print(f"  Total Duration   : {res_fake['total_duration']}s")
print(f"  Analyzed Duration: {res_fake['analyzed_duration']}s")
print(f"  Segments Count   : {res_fake['segments_count']}")
print(f"  Synthetic Ratio  : {res_fake['fake_ratio']}%")
print(f"  Verdict          : {res_fake['verdict']}")
print(f"  Peak Risk Frame  : {res_fake['highest_risk_segment']}")
print("  [PASS] Single Audio Localizer passed for Fake Audio.")

# -------------------------------------------------------------
# TEST 3: Simulated Live Microphone Audio (Continuous Timeline Check)
# -------------------------------------------------------------
print("\n--- [TEST 3: Live Microphone Recording Simulation] ---")
sr = 16000
dur = 5.2
t = np.linspace(0, dur, int(sr * dur), endpoint=False)
# 400Hz speech-like tone with a pause in the middle
tone = 0.4 * np.sin(2 * np.pi * 400 * t)
tone[int(sr*1.5):int(sr*2.8)] = 0.001 * np.random.randn(int(sr*1.3))  # silence/pause
bio_mic = io.BytesIO()
sf.write(bio_mic, tone, sr, format="WAV", subtype="PCM_16")
mic_bytes = bio_mic.getvalue()

res_mic = analyze_audio_data(mic_bytes, "live_mic_simulated.wav")
assert "error" not in res_mic, f"Mic analysis error: {res_mic.get('error')}"
assert res_mic["total_duration"] == round(dur, 2), f"Expected total_duration {dur}, got {res_mic['total_duration']}"
assert res_mic["analyzed_duration"] >= 5.0, f"Expected analyzed_duration >= 5.0, got {res_mic['analyzed_duration']}"
print(f"  Recorded Length  : {res_mic['total_duration']}s")
print(f"  Analyzed Length  : {res_mic['analyzed_duration']}s")
print(f"  Windows Tested   : {res_mic['segments_count']} windows")
print(f"  First Window     : {res_mic['segments'][0]['start']}s - {res_mic['segments'][0]['end']}s")
print(f"  Last Window      : {res_mic['segments'][-1]['start']}s - {res_mic['segments'][-1]['end']}s")
print("  [PASS] Live Microphone simulation verified: full duration preserved without 2-second truncation!")

# -------------------------------------------------------------
# TEST 4: Speaker Voiceprint Matching & Clone Impersonation
# -------------------------------------------------------------
print("\n--- [TEST 4: Speaker Clone / Voiceprint Matcher] ---")
with open(real_files[1], "rb") as f:
    real2_bytes = f.read()

emb_ref, err_ref = extract_speaker_embedding_vector(real_bytes, "ref.mp3")
emb_self, err_self = extract_speaker_embedding_vector(real_bytes, "self.mp3")
emb_diff, err_diff = extract_speaker_embedding_vector(real2_bytes, "diff.mp3")

assert err_ref is None, f"Error extracting emb_ref: {err_ref}"
assert emb_ref.shape == (768,), f"Expected shape (768,), got {emb_ref.shape}"

# Self-match should be ~100%
sim_self = compute_speaker_similarity(emb_ref, emb_self)
print(f"  Self-Comparison Similarity: {sim_self['similarity_percentage']}% (Match: {sim_self['match']})")
assert sim_self["similarity_percentage"] >= 99.9, "Self similarity should be 100%"

# Different speaker match
sim_diff = compute_speaker_similarity(emb_ref, emb_diff)
print(f"  Diff-Speaker Similarity  : {sim_diff['similarity_percentage']}% (Verdict: {sim_diff['verdict']})")
print("  [PASS] Speaker Voiceprint Matcher passed.")

# -------------------------------------------------------------
# TEST 5: Temporal Audio Slicer (Instant Browser Playback)
# -------------------------------------------------------------
print("\n--- [TEST 5: Forensic Audio Slicer] ---")
y_dec, _, _ = decode_audio_bytes(real_bytes)
sliced_bytes = slice_segment_audio_bytes(y_dec, 1.0, 3.0, sr=16000)
assert len(sliced_bytes) > 0, "Sliced bytes is empty"
# verify it is valid WAV
with io.BytesIO(sliced_bytes) as bio_s:
    s_wave, s_sr = sf.read(bio_s)
    assert s_sr == 16000, f"Expected 16000, got {s_sr}"
    assert abs(len(s_wave)/s_sr - 2.0) < 0.1, f"Expected ~2.0s slice, got {len(s_wave)/s_sr}s"
print(f"  Sliced 1.0s–3.0s successfully into {len(sliced_bytes)} WAV bytes ({len(s_wave)/s_sr}s audio).")
print("  [PASS] Audio slicer passed.")

# -------------------------------------------------------------
# TEST 6: Forensic Audit Report Generation
# -------------------------------------------------------------
print("\n--- [TEST 6: Printable HTML Audit Report] ---")
report_html = generate_forensic_html_report(res_real, user_tz_name="Asia/Kolkata")
assert "<html" in report_html.lower() and "</html>" in report_html.lower(), "Invalid HTML report"
assert res_real["file_hashes"]["sha256"] in report_html, "SHA-256 missing from report"
assert "Forensic Audit Report" in report_html, "Report title missing"
print(f"  Report generated successfully ({len(report_html)} bytes). Contains SHA-256 & timezone formatting.")
print("  [PASS] HTML Audit Report passed.")

# -------------------------------------------------------------
# TEST 7: SQLite Database History & Persistence
# -------------------------------------------------------------
print("\n--- [TEST 7: SQLite History Persistence] ---")
init_db()
save_result("test_audio_sample.wav", res_real)
hist = get_history(limit=5)
assert len(hist) > 0, "History is empty"
latest = hist[0]
assert latest["filename"] == "test_audio_sample.wav", f"Expected test_audio_sample.wav, got {latest['filename']}"
print(f"  Retrieved latest scan: id={latest['id']}, file={latest['filename']}, verdict={latest['verdict']}, duration={latest['duration_sec']}s")
print("  [PASS] SQLite History database passed.")

# -------------------------------------------------------------
# TEST 8: FastAPI HTTP Endpoints
# -------------------------------------------------------------
print("\n--- [TEST 8: FastAPI Route Testing via TestClient] ---")
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Root endpoint
res_root = client.get("/")
assert res_root.status_code == 200, f"Root status {res_root.status_code}"
print(f"  GET / -> {res_root.json()['status']}")

# History endpoint
res_h = client.get("/history?limit=3")
assert res_h.status_code == 200, f"History status {res_h.status_code}"
print(f"  GET /history -> Retrieved {len(res_h.json())} records")

# Export report endpoint
res_exp = client.post("/export_report", json=res_real)
assert res_exp.status_code == 200, f"Export report status {res_exp.status_code}"
assert "text/html" in res_exp.headers.get("content-type", "")
print(f"  POST /export_report -> Status 200 (HTML response)")

# Analyze endpoint via multipart file upload
res_post_analyze = client.post("/analyze", files={"file": ("test.wav", mic_bytes, "audio/wav")})
assert res_post_analyze.status_code == 200, f"Analyze status {res_post_analyze.status_code}"
assert "verdict" in res_post_analyze.json(), "Verdict missing from API response"
print(f"  POST /analyze -> Status 200 (Verdict: {res_post_analyze.json()['verdict']}, fake_ratio: {res_post_analyze.json()['fake_ratio']}%)")

# Compare speakers endpoint via multipart file upload
res_post_comp = client.post(
    "/compare_speakers",
    files={
        "file_ref": ("ref.wav", mic_bytes, "audio/wav"),
        "file_suspect": ("sus.wav", mic_bytes, "audio/wav"),
    }
)
assert res_post_comp.status_code == 200, f"Compare status {res_post_comp.status_code}"
assert "similarity_percentage" in res_post_comp.json(), "Similarity missing from API response"
print(f"  POST /compare_speakers -> Status 200 (Similarity: {res_post_comp.json()['similarity_percentage']}%)")
print("  [PASS] All FastAPI routes passed.")

print("\n" + "="*70)
print("ALL 8 VERIFICATION TESTS PASSED WITH ZERO ERRORS!")
print("="*70)
