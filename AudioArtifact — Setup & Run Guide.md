# AudioArtifact — Setup & Run Guide

This guide explains the complete step-by-step process to run the project from scratch.

Run every command from the project root folder (the folder where this README is located).

## 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add the Dataset

Place genuine human voice recordings (`.wav` / `.mp3`) in the `data/real/` folder.

Place AI-generated / deepfake voice recordings in the `data/fake/` folder.

## 3. Extract Features

```bash
python feature_extraction.py
```

This will generate `features.csv` in the project root folder.

## 4. Train the Model

```bash
python train_model.py
```

This will generate `model/deepfake_detector.pkl`, and the accuracy report will be displayed in the console.

## 5. Start the Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Check the backend by opening `http://127.0.0.1:8000` in your browser.

You should see:

```json
{"status": "AudioArtifact backend running"}
```

## 6. Start the Frontend (in a new terminal)

```bash
streamlit run frontend/app.py
```

The browser will open automatically. Upload an audio file there and click **"Analyze"**.

## Folder Structure

```text
AudioArtifact/
├── data/
│   ├── real/                    <- Put human voice files here
│   └── fake/                    <- Put AI/fake voice files here
├── model/
│   └── deepfake_detector.pkl    <- Generated after running train_model.py
├── backend/
│   ├── main.py                  <- FastAPI server
│   └── db.py                    <- SQLite history
├── frontend/
│   └── app.py                   <- Streamlit dashboard
├── feature_extraction.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Common Issues

- **"No module named librosa"** → The virtual environment is not activated, or `pip install -r requirements.txt` has not been run.
- **"Model not loaded"** error from the backend → Run `train_model.py` first.
- **Unable to connect to the backend** → Make sure the backend is running in the terminal and port `8000` is available.