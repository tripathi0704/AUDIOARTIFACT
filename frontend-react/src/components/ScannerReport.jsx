import { useEffect, useRef, useState } from "react";

const CIRC = 251.2;

function formatTime(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export default function ScannerReport({ result }) {
  const { filename, total_duration, fake_seconds, segments } = result;
  const trackRef = useRef(null);
  const [revealed, setRevealed] = useState(false);
  const [scanOn, setScanOn] = useState(false);
  const [scanLeft, setScanLeft] = useState("0%");
  const [tooltip, setTooltip] = useState(null); // {x, seg}

  const avgConf = Math.round(
    segments.reduce((sum, s) => sum + s.confidence, 0) / Math.max(segments.length, 1) * 10
  ) / 10;

  const verdictClass =
    fake_seconds === 0 ? "is-real" : fake_seconds >= total_duration ? "is-fake" : "is-mixed";
  const verdictLabel =
    fake_seconds === 0 ? "Human-verified" : fake_seconds >= total_duration ? "Fully synthetic" : "Mixed signal detected";

  const gaugeOffset = CIRC * (1 - (revealed ? avgConf : 0) / 100);
  const gaugeColor = avgConf >= 70 ? "var(--real)" : "var(--amber)";

  useEffect(() => {
    setRevealed(false);
    setScanOn(false);
    setScanLeft("0%");
    const t1 = setTimeout(() => {
      setScanOn(true);
      setScanLeft("100%");
      setRevealed(true);
    }, 250);
    const t2 = setTimeout(() => setScanOn(false), 1850);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [result]);

  const rulerSteps = 6;
  const rulerMarks = Array.from({ length: rulerSteps + 1 }, (_, i) => (total_duration / rulerSteps) * i);

  return (
    <div className="shell">
      <div className="topbar">
        <div>
          <div className="fname">{filename}</div>
          <div className="fsub mono">{Math.round(total_duration)}s scanned</div>
        </div>
        <div className={`pill ${verdictClass}`}><i></i><span>{verdictLabel}</span></div>
      </div>

      <div className="report-body">
        <div className="canvas">
          <div className="baseline"></div>
          <div
            className="track"
            ref={trackRef}
            onMouseMove={(e) => {
              const rect = trackRef.current.getBoundingClientRect();
              const idx = Math.floor(((e.clientX - rect.left) / rect.width) * segments.length);
              const seg = segments[Math.min(Math.max(idx, 0), segments.length - 1)];
              setTooltip({ x: e.clientX - rect.left, seg });
            }}
            onMouseLeave={() => setTooltip(null)}
          >
            {segments.map((seg, idx) => (
              <div
                key={idx}
                className={`bar ${revealed ? "revealed" : ""} ${seg.label_code === 0 ? "real" : "fake"}`}
                style={{
                  height: `${28 + (seg.confidence % 40)}px`,
                  transitionDelay: `${(idx / segments.length) * 1500}ms`,
                }}
              />
            ))}
          </div>
          <div className="scanline" style={{ left: scanLeft, opacity: scanOn ? 1 : 0 }}></div>
          {tooltip && (
            <div className="tooltip" style={{ left: tooltip.x, top: 0 }}>
              {tooltip.seg.label_code === 0 ? "Human-verified" : "AI-synthesized"}
              <div className="tt-sub">
                {tooltip.seg.start.toFixed(1)}s–{tooltip.seg.end.toFixed(1)}s · {tooltip.seg.confidence}%
              </div>
            </div>
          )}
        </div>
        <div className="ruler mono">
          {rulerMarks.map((t, i) => <span key={i}>{formatTime(t)}</span>)}
        </div>
      </div>

      <div className="bottom">
        <div className="legend">
          <span><i style={{ background: "var(--real)" }}></i>Human-verified</span>
          <span><i style={{ background: "var(--fake)" }}></i>AI-synthesized</span>
        </div>
        <div className="gauge-wrap">
          <svg viewBox="0 0 90 90">
            <circle className="gauge-bg" cx="45" cy="45" r="40" />
            <circle
              className="gauge-fg"
              cx="45" cy="45" r="40"
              style={{ stroke: gaugeColor, strokeDashoffset: gaugeOffset }}
            />
          </svg>
          <div>
            <div className="gauge-num">{revealed ? avgConf : 0}%</div>
            <div className="gauge-cap">confidence</div>
          </div>
        </div>
      </div>
    </div>
  );
}
