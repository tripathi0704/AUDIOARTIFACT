# AudioArtifact — Setup & Run Guide

Yeh guide poore project ko scratch se chalane ka step-by-step process hai.
Har command project ke root folder (jaha ye README hai) se chalao.

## 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows par: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Dataset Daalo

`data/real/` folder mein genuine human voice recordings daalo (.wav / .mp3).
`data/fake/` folder mein AI-generated / deepfake voice recordings daalo.

Free datasets kaha se milenge:
- ASVspoof 2019 / 2021 — https://www.asvspoof.org/
- Fake-or-Real (FoR) Dataset — Kaggle par search karo
- WaveFake — https://github.com/RUB-SysSec/WaveFake

Kam se kam 50-100 files har folder mein rakho taaki model theek se train ho.

## 3. Features Extract Karo

```bash
python feature_extraction.py
```

Isse `features.csv` ban jayegi (root folder mein).

## 4. Model Train Karo

```bash
python train_model.py
```

Isse `model/deepfake_detector.pkl` ban jayega, aur accuracy report console par dikhegi.

## 5. Backend Start Karo

```bash
uvicorn backend.main:app --reload --port 8000
```

Check karo: browser mein `http://127.0.0.1:8000` khol kar dekho — `{"status": "AudioArtifact backend running"}` dikhna chahiye.

## 6. Frontend Start Karo (naye terminal mein)

```bash
streamlit run frontend/app.py
```

Browser automatically khul jayega — wahan file upload karke "Analyze" click karo.

## Folder Structure

```
AudioArtifact/
├── data/
│   ├── real/              <- yaha human voice files daalo
│   └── fake/               <- yaha AI/fake voice files daalo
├── model/
│   └── deepfake_detector.pkl   <- train_model.py ke baad yaha banega
├── backend/
│   ├── main.py              <- FastAPI server
│   └── db.py                <- SQLite history
├── frontend/
│   └── app.py                <- Streamlit dashboard
├── feature_extraction.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Common Issues

- **"No module named librosa"** → virtual environment activate nahi hai, ya `pip install -r requirements.txt` nahi chala.
- **"Model not loaded"** error backend se → `train_model.py` pehle chalao.
- **Backend se connection nahi ho raha** → check karo backend terminal mein chal raha hai aur port 8000 free hai.
