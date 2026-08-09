# AudioArtifact — React Frontend

Ye Streamlit dashboard ka replacement hai — bilkul same scanner UI, lekin React + Vite mein.
Backend (FastAPI) bilkul waisa hi rahega, koi change nahi.

## Pehli Baar Setup (sirf ek baar)

1. **Node.js install karo** (agar nahi hai) — https://nodejs.org se LTS version download karo
2. Terminal mein is folder ke andar aao aur dependencies install karo:

```bash
cd frontend-react
npm install
```

## Har Baar Chalane Ke Liye

**Terminal 1 — Backend (waisa hi rahega):**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — React frontend (naya):**
```bash
cd frontend-react
npm run dev
```

Browser mein `http://localhost:5173` khol lo — waha dashboard dikhega.

## Folder Structure

```
frontend-react/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx           <- entry point
    ├── App.jsx             <- main component, backend se connect karta hai
    ├── index.css            <- poora dark theme yahin hai
    └── components/
        ├── UploadZone.jsx     <- drag-and-drop uploader + audio player
        ├── ScannerReport.jsx  <- timeline, verdict, gauge, tooltip
        └── SignatureGraph.jsx <- 3D spectrogram graph (Plotly)
```

## Kaise Kaam Karta Hai

1. `UploadZone` file uploader dikhata hai — file select hone par `App.jsx` ko batata hai
2. "Analyze" click karne par `App.jsx` file ko `fetch()` se FastAPI backend ke `/analyze` endpoint par bhejta hai
3. Backend response (segments, verdict, spectrogram data) wapas aata hai
4. `ScannerReport` us data se timeline bar, verdict pill, aur confidence gauge banata hai
5. `SignatureGraph` backend se aayi spectrogram values se 3D graph banata hai (Plotly)

## Production Build (deploy karne ke liye, optional)

```bash
npm run build
```
Isse `dist/` folder banega jo kisi bhi static hosting (Vercel, Netlify) par daal sakte ho.
Backend URL (`App.jsx` mein `BACKEND_URL`) ko tab apne live backend ke URL se replace karna hoga.
