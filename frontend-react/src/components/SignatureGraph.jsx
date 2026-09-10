import Plot from "react-plotly.js";

export default function SignatureGraph({ spectrogram }) {
  if (!spectrogram) return null;

  return (
    <Plot
      data={[
        {
          type: "surface",
          z: spectrogram.z,
          x: spectrogram.time,
          y: spectrogram.mel,
          showscale: false,
          colorscale: [
            [0, "#0A0C0B"],
            [0.4, "#123024"],
            [0.7, "#1e6b4c"],
            [1, "#3ECF8E"],
          ],
        },
      ]}
      layout={{
        autosize: true,
        height: 420,
        margin: { l: 0, r: 0, t: 10, b: 0 },
        paper_bgcolor: "rgba(0,0,0,0)",
        scene: {
          xaxis: { title: "", showbackground: false, color: "#5E6660" },
          yaxis: { title: "", showbackground: false, color: "#5E6660" },
          zaxis: { title: "", showbackground: false, color: "#5E6660" },
          bgcolor: "rgba(0,0,0,0)",
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
    />
  );
}
