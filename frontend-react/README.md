# AudioArtifact — React Frontend

This React + Vite frontend is a replacement for the Streamlit dashboard. It provides the same scanner UI, while the backend (FastAPI) remains unchanged.

## First-time setup (one-time)

1. Install Node.js (LTS) from https://nodejs.org if you don't have it installed.
2. From this folder, install dependencies:

```bash
cd frontend-react
npm install
```

## Running (every time)

Terminal 1 — Backend (unchanged):
```bash
uvicorn backend.main:app --reload --port 8000
```

Terminal 2 — React frontend:
```bash
cd frontend-react
npm run dev
```

Open `http://localhost:5173` in your browser to view the dashboard.

## Folder Structure

```
frontend-react/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx           <- entry point
    ├── App.jsx            <- main component that connects to the backend
    ├── index.css          <- site-wide dark theme
    └── components/
        ├── UploadZone.jsx     <- drag-and-drop uploader + audio player
        ├── ScannerReport.jsx  <- timeline, verdict, gauge, tooltip
        └── SignatureGraph.jsx <- 3D spectrogram graph (Plotly)
```

## How it works

1. `UploadZone` displays a file uploader — when a file is selected it notifies `App.jsx`.
2. Clicking the "Analyze" button sends the file to the FastAPI backend `/analyze` endpoint via `fetch()`.
3. The backend returns response data (segments, verdict, spectrogram values).
4. `ScannerReport` renders the timeline bar, verdict pill, and confidence gauge from that data.
5. `SignatureGraph` uses the spectrogram values to render a 3D graph with Plotly.

## Production build (for deployment)

```bash
npm run build
```

This produces a `dist/` folder that can be deployed to any static hosting service (Vercel, Netlify, etc.).
Remember to replace the `BACKEND_URL` in `App.jsx` with your live backend URL when deploying.
