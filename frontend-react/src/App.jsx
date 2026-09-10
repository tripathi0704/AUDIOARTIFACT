import { useState } from "react";
import UploadZone from "./components/UploadZone.jsx";
import LiveVisualizer from "./components/LiveVisualizer.jsx";
import ScannerReport from "./components/ScannerReport.jsx";
import SignatureGraph from "./components/SignatureGraph.jsx";
import useAudioAnalyser from "./hooks/useAudioAnalyser.js";

const BACKEND_URL = "http://127.0.0.1:8000/analyze";

export default function App() {
  const [file, setFile] = useState(null);
  const [audioEl, setAudioEl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyser = useAudioAnalyser(audioEl);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(BACKEND_URL, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(`Could not reach the scanner service: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="noise-bg"></div>
      <div className="glow"></div>

      <div className="app-shell">
        <div className="brand-row">
          <div className="brand"><span className="px"></span>AudioArtifact</div>
          <div className="status-tag"><i></i>SCANNER ONLINE</div>
        </div>

        <h1 className="hero-title">Every voice leaves a signature.</h1>
        <p className="hero-sub">
          Drop a clip below to scan it segment by segment and see exactly
          where — if anywhere — it turns synthetic.
        </p>

        <UploadZone
          file={file}
          onFileSelected={(f) => { setFile(f); setResult(null); setError(null); setAudioEl(null); }}
          onClear={() => { setFile(null); setResult(null); setError(null); setAudioEl(null); }}
          onAudioReady={setAudioEl}
        />

        {file && <LiveVisualizer audioEl={audioEl} analyser={analyser} />}

        {file && (
          <button className="analyze-btn" onClick={handleAnalyze} disabled={loading}>
            {loading ? (<><span className="spinner"></span>Scanning...</>) : "Analyze"}
          </button>
        )}

        {error && <div className="error-box">{error}</div>}

        {result && (
          <>
            <div style={{ marginTop: 32 }}>
              <ScannerReport result={result} audioEl={audioEl} analyser={analyser} />
            </div>

            <div className="section-label">Voice Signature</div>
            <SignatureGraph spectrogram={result.spectrogram} />
          </>
        )}
      </div>
    </>
  );
}
