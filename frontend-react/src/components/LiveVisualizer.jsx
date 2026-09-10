import { useEffect, useRef } from "react";

export default function LiveVisualizer({ audioEl, analyser }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!audioEl || !analyser) return;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    const canvas = canvasRef.current;
    const c2d = canvas.getContext("2d");

    const resize = () => {
      canvas.width = canvas.clientWidth * window.devicePixelRatio;
      canvas.height = canvas.clientHeight * window.devicePixelRatio;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      const w = canvas.width, h = canvas.height;
      c2d.clearRect(0, 0, w, h);

      const barCount = dataArray.length;
      const gap = 3 * window.devicePixelRatio;
      const barWidth = w / barCount - gap;

      for (let i = 0; i < barCount; i++) {
        const value = dataArray[i] / 255;
        const barHeight = Math.max(value * h, 3);
        const x = i * (barWidth + gap);
        const y = (h - barHeight) / 2;

        const gradient = c2d.createLinearGradient(0, y, 0, y + barHeight);
        gradient.addColorStop(0, "#5EEAD4");
        gradient.addColorStop(1, "#3ECF8E");
        c2d.fillStyle = gradient;
        c2d.fillRect(x, y, barWidth, barHeight);
      }
    };

    const startLoop = () => { if (!rafRef.current) draw(); };
    const stopLoop = () => {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      c2d.clearRect(0, 0, canvas.width, canvas.height);
    };

    audioEl.addEventListener("play", startLoop);
    audioEl.addEventListener("pause", stopLoop);
    audioEl.addEventListener("ended", stopLoop);
    if (!audioEl.paused) startLoop();

    return () => {
      window.removeEventListener("resize", resize);
      audioEl.removeEventListener("play", startLoop);
      audioEl.removeEventListener("pause", stopLoop);
      audioEl.removeEventListener("ended", stopLoop);
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [audioEl, analyser]);

  return (
    <div className="live-visualizer">
      <canvas ref={canvasRef}></canvas>
    </div>
  );
}
