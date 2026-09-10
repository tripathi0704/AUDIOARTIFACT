import { useEffect, useRef, useState } from "react";

/**
 * Sets up exactly one Web Audio AnalyserNode per <audio> element and
 * shares it with any component that needs live frequency data
 * (the upload-stage visualizer AND the post-analysis result timeline).
 */
export default function useAudioAnalyser(audioEl) {
  const audioCtxRef = useRef(null);
  const [analyser, setAnalyser] = useState(null);

  useEffect(() => {
    if (!audioEl) {
      setAnalyser(null);
      return;
    }

    if (!audioCtxRef.current) {
      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = audioCtxRef.current;

    const node = ctx.createAnalyser();
    node.fftSize = 128;
    node.smoothingTimeConstant = 0.75;

    let source;
    try {
      source = ctx.createMediaElementSource(audioEl);
      source.connect(node);
      node.connect(ctx.destination);
      setAnalyser(node);
    } catch (e) {
      // this audio element already has a source attached - nothing to do
      setAnalyser(null);
      return;
    }

    const resumeCtx = () => ctx.resume();
    audioEl.addEventListener("play", resumeCtx);

    return () => {
      audioEl.removeEventListener("play", resumeCtx);
      try { source.disconnect(); node.disconnect(); } catch (e) {}
      setAnalyser(null);
    };
  }, [audioEl]);

  return analyser;
}
