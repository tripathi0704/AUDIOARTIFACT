# ◈ AudioArtifact v2.0 — Timeline-Based Deepfake Audio Localizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://audioartifact.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AudioArtifact is an advanced timeline-based forensic system for detecting and localizing synthetic speech segments in audio recordings. It leverages **Silero Voice Activity Detection (VAD)**, **50% overlapping sliding windows**, **Microsoft WavLM 768-dimensional deep acoustic embeddings**, **calibrated XGBoost continuous probability modeling**, and an **interactive Plotly + Streamlit visualization dashboard**.

### 🌟 Enterprise Forensic Capabilities (v2.0 Extended):
1. **◈ Single Audio Localizer**:
   - **Segment-by-Segment Click-to-Play**: Instant audio player for each 2-second suspect window to hear synthetic glitches in isolation.
   - **Cryptographic Chain-of-Custody**: SHA-256 and MD5 bitstream checksum verification.
   - **Biological & Acoustic Artifacts**: High-frequency spectral rolloff (vocoder cutoff), zero-crossing rate, dynamic range, and vocal pitch inflection.
   - **1-Click Forensic Audit Certificate**: Printable ISO-style HTML/PDF forensic report download.
   - **Sensitivity Calibration**: Adjustable AI decision threshold (30%–75%).
2. **🎙️ Live Mic Voice Test**: Real-time microphone audio recording and instant deepfake screening.
3. **👥 Speaker Clone / Voiceprint Matcher**: WavLM 768-dim cosine similarity cross-matching between a Reference Voice and a Suspect Audio to detect impersonation and voice cloning attacks.
4. **📁 Batch Forensic Scanner**: Multi-file bulk processing, progress tracking, and CSV audit export.
5. **📜 Session History & Audit Log**: Searchable, filterable scan records with side-by-side **Inspect ◈** and **💬 Chat** buttons to reinspect audio scans and resume full AI copilot chat histories anytime.
6. **🤖 Dedicated Forensic AI Copilot & Modal Dialog**:
   - **Isolated Modal Chatbox**: High-resolution dialog box for in-depth, multi-turn forensic dialogue per audio recording.
   - **Multi-File Session Memory**: Each uploaded recording remembers its own complete conversation history during the browser session.
   - **Sub-Second Low-Latency Model Chain**: High-speed neural inference pipeline (`gemini-3.5-flash-lite`, `gemini-flash-lite-latest`) with automatic zero-downtime failover.
   - **Audio Waveform Grounding & Anti-Hallucination Guardrails**: Listens to actual waveforms; accurately distinguishes musical instruments (e.g., Veena, Sitar, piano) and ambient sounds from human voices to prevent speech/scam hallucinations.
   - **Scam & Coercion Threat Intelligence**: Semantic screening for financial extortion, fake arrest/kidnap coercion, OTP phishing, and malicious impersonation.
7. **⚖️ Court-Ready 2-Page Cyber Crime FIR Petition & Sec. 63 BSA Certificate (PDF)**:
   - **Direct Print & Submit (Zero Lawyer Consultation Needed)**: Ready for immediate printing and submission to any Cyber Crime Police Station (SHO) or judicial magistrate.
   - **New Criminal Laws Compliant**: Filed under **Section 173 BNSS, 2023** (formerly Sec. 154 CrPC) with dual penal provisions under **Bharatiya Nyaya Sanhita (BNS), 2023** (Sec. 318(4), 319(2), 336(3)) and **IT Act, 2000** (Sec. 66D, 66E).
   - **Mandatory Electronic Evidence Certificate (Annexure - A)**: Executed under **Section 63 of Bharatiya Sakshya Adhiniyam (BSA), 2023** (formerly Sec. 65B(4) Indian Evidence Act) compliant with Supreme Court landmark precedent (*Arjun Panditrao Khotkar (2020) 7 SCC 1*).
   - **Flawless 2-Page Turnkey Layout**: Exactly 2 pages without text truncation or cell overlapping; includes 1-click pre-fill forms and clean pen-fillable blanks.


---

## 🌐 Live Web App (Direct Access)

> [!TIP]
> You can try the live interactive forensic detector directly in your browser without installing anything:
> 👉 **[https://audioartifact.streamlit.app](https://audioartifact.streamlit.app)**

Upload any suspicious audio file (`.wav` or `.mp3`) to test the real-time deepfake timeline detector instantly!

---

## ⚠️ Important: Copying this Project to Another Computer

> [!CAUTION]
> **DO NOT copy the `venv/` folder to a new computer!**
> 
> In Python, a virtual environment (`venv`) contains **hardcoded absolute paths** pointing to the Python installation of the original machine (e.g., `C:\Users\shubh\...`). 
> 
> If you copy the `venv/` folder to a different PC and try to run the application, you will immediately encounter errors like:
> - `Fatal error in launcher: Unable to create process using 'C:\Users\...'`
> - `The system cannot find the path specified.`
> - Command windows that instantly open and close.
>
> **Solution**:
> 1. If you copied the entire folder including `venv/`, **DELETE the `venv` folder** on your new computer.
> 2. Run `setup.bat` (or follow the setup instructions below) to generate a fresh, clean environment tailored to your new computer.

---

## 📋 System Prerequisites

Before setting up the project, make sure your computer has the following installed:

### 1. Python (Version 3.10, 3.11, or 3.12 — 64-bit)
- Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- ⚠️ **CRITICAL STEP DURING INSTALLATION**:
  Ensure you check the box:
  ```text
  [✔] Add python.exe to PATH
  ```
  *(If you forget this, typing `python` in your terminal will not work).*
- Verify in your terminal:
  ```bash
  python --version
  ```

### 2. FFmpeg (Required for MP3 & compressed audio decoding)
`librosa` and `audioread` require FFmpeg to load `.mp3`, `.m4a`, and `.ogg` files. *(Uncompressed `.wav` files work out of the box).*
- **Windows (easiest via winget)**:
  ```powershell
  winget install Gyan.FFmpeg
  ```
  *Alternatively, download the release build from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) and add its `bin` folder to your System PATH.*
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

---

## 🚀 Method 1: Automated 1-Click Setup (Recommended for Windows)

We provide pre-built automated scripts that handle everything for you:

### Step 1: Run Environment Setup
Double-click `setup.bat` (or run it in Command Prompt):
```cmd
setup.bat
```
What `setup.bat` does automatically:
1. Detects and verifies your Python installation and PATH.
2. Checks if an old/broken `venv` was copied from another computer and cleans it.
3. Creates a fresh virtual environment in `.\venv`.
4. Upgrades `pip`.
5. Installs the lightweight CPU PyTorch & torchaudio package.
6. Installs all required packages from `requirements.txt`.
7. Validates all core imports (`FastAPI`, `Streamlit`, `Librosa`, `XGBoost`, `Torch`, `Silero VAD`).

### Step 2: Launch the System
Once `setup.bat` finishes, double-click `start.bat`:
```cmd
start.bat
```
This automatically launches:
- **Backend (FastAPI)** in one window at: `http://127.0.0.1:8000`
- **Frontend (Streamlit)** in a second window at: `http://localhost:8501`

---

## 🛠️ Method 2: Manual Step-by-Step Setup

If you prefer setting up manually in your terminal, follow these steps:

### 1. Open Terminal in Project Root
Make sure your terminal is opened in the project root directory (where `requirements.txt` is located):
```bash
cd /d D:\audioartifact    # Windows CMD
```

### 2. Create Virtual Environment
```bash
# Windows / macOS / Linux:
python -m venv venv
```

### 3. Activate Virtual Environment
- **Windows (Command Prompt - CMD)**:
  ```cmd
  venv\Scripts\activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If you get a script execution policy error, see the Troubleshooting section below).*
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### 4. Upgrade Pip
```bash
python -m pip install --upgrade pip
```

### 5. Install PyTorch (CPU-Optimized)
Installing PyTorch CPU directly first saves 2+ GB of unnecessary GPU CUDA downloads and avoids download timeouts:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 6. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 7. Verify the Environment
Run this verification command to confirm all modules are properly installed:
```bash
python -c "import fastapi, uvicorn, streamlit, librosa, xgboost, torch, silero_vad; print('>>> All core dependencies loaded successfully!')"
```

---

## ▶️ Running the Application Manually

If you prefer running without `start.bat`, open **two separate terminal windows** and activate your venv in both:

### Terminal 1: Backend Service (FastAPI)
```bash
# Activate venv:
# Windows CMD: venv\Scripts\activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate

uvicorn backend.main:app --reload --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`

### Terminal 2: Frontend Dashboard (Streamlit)
```bash
# Activate venv:
# Windows CMD: venv\Scripts\activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate

streamlit run frontend/app.py
```
- Interactive Dashboard UI: `http://localhost:8501`

---

## 🤖 Model & Dataset Information

### Pre-trained Model Included
The pre-trained XGBoost detector model is already included in this repository:
```text
model/deepfake_detector.pkl
```
You **do not need to retrain the model** to run the app. It works immediately out of the box!

### Retraining on Custom Data (Optional)
If you want to train on your own audio recordings:
1. Place genuine recordings in `data/real/` (.wav or .mp3).
2. Place AI-generated recordings in `data/fake/` (.wav or .mp3).
3. Extract features:
   ```bash
   python feature_extraction.py
   ```
   *(Options: `--limit 150` for balanced subset, or `--all` for all files)*.
4. Train the XGBoost model:
   ```bash
   python train_model.py
   ```
   This will output Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC, updating `model/deepfake_detector.pkl`.

---

## 📂 Project Structure

```text
AudioArtifact/
├── backend/
│   ├── main.py                  <- FastAPI async API service (/analyze, /history)
│   ├── ai_copilot.py            <- Gemini Forensic Copilot, XAI explanations, scam detection
│   ├── forensic_utils.py        <- SHA-256 hashes, HTML audit report, 2-page FIR & Sec 63 BSA PDF
│   ├── models.py                <- WavLM 768-dim embeddings & XGBoost inference pipeline
│   ├── vad_utils.py             <- Silero VAD + energy-based silence filtering
│   └── db.py                    <- SQLite persistence & scan history
├── frontend/
│   └── app.py                   <- Streamlit dashboard with Plotly probability curve & AI chat modal
├── data/
│   ├── real/                    <- Human genuine audio dataset (.wav / .mp3)
│   └── fake/                    <- Synthetic / cloned audio dataset (.wav / .mp3)
├── model/
│   └── deepfake_detector.pkl    <- Pre-trained calibrated XGBoost model
├── feature_extraction.py        <- 60-feature extractor with VAD and 50% overlap
├── train_model.py               <- XGBoost training script with ROC-AUC evaluation
├── requirements.txt             <- Project Python package dependencies
├── setup.bat                    <- 1-click automated environment setup script (Windows)
├── start.bat                    <- 1-click dual-server launcher (Windows)
├── history.db                   <- SQLite forensic scan history database
└── README.md                    <- Engineering & setup documentation
```

---

## ⚖️ Legal Admissibility & Court Protocol (BNS, BNSS 2023 & Sec. 63 BSA)

AudioArtifact generates **100% turnkey, court-ready legal documentation** that citizens, forensic examiners, and cybercrime investigators can directly print and submit to police stations or courts without requiring prior legal drafting.

### Statutory Criminal Provisions Invoked:
1. **Section 173 of Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023** *(formerly Section 154 CrPC)*:
   - Mandatory registration of First Information Report (FIR) for cognizable cyber fraud.
2. **Information Technology Act, 2000**:
   - **Section 66D**: Cheating by personation using computer resources (imprisonment up to 3 years + fine).
   - **Section 66E**: Privacy violations and non-consensual capture/transmission of synthetic likeness.
3. **Bharatiya Nyaya Sanhita (BNS), 2023**:
   - **Section 319(2)** *(formerly IPC 419)*: Punishment for cheating by personation.
   - **Section 318(4)** *(formerly IPC 420)*: Cheating and dishonestly inducing delivery of property/funds.
   - **Section 336(3) & 340(2)** *(formerly IPC 468/471)*: Forgery of electronic records for cheating.

### Mandatory Electronic Evidence Certificate (Annexure - A):
Under Indian evidence law, digital audio recordings are strictly inadmissible unless accompanied by a statutory certificate of authenticity:
- Executed under **Section 63 of Bharatiya Sakshya Adhiniyam (BSA), 2023** *(formerly Section 65B(4) of the Indian Evidence Act, 1872)*.
- Fully adheres to the mandatory legal safeguards laid down by the Hon'ble Supreme Court of India in ***Arjun Panditrao Khotkar v. Kailash Kushanrao Gorantyal (2020) 7 SCC 1***.
- Certified SHA-256 bitstream checksum and MD5 signature permanently bind the physical paper complaint to the digital audio bitstream.

---

## ❓ Troubleshooting & FAQs

### 1. `Fatal error in launcher: Unable to create process using '...'`
- **Cause**: The `venv` directory was copied from another computer. The scripts inside `venv\Scripts\` still point to the previous machine's Python directory.
- **Fix**: 
  1. Delete the `venv` folder completely:
     ```cmd
     rmdir /s /q venv
     ```
  2. Run `setup.bat` (or recreate with `python -m venv venv`).

---

### 2. PowerShell Error: `File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system`
- **Cause**: Windows PowerShell restricts script execution by default.
- **Fix**: Open PowerShell as Administrator or as your current user and run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
  Then activate again:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *Alternatively, switch to standard Command Prompt (CMD) where this restriction does not apply.*

---

### 3. `'python' is not recognized as an internal or external command`
- **Cause**: Python is not in your system environment PATH.
- **Fix**:
  1. Re-run the Python installer.
  2. Select **Modify** or **Repair**.
  3. Ensure **"Add python.exe to PATH"** is checked.
  4. Restart your terminal window.

---

### 4. `NoBackendError` or `AudioreadError` when analyzing `.mp3` files
- **Cause**: FFmpeg is missing on Windows. `librosa` cannot decode compressed audio without FFmpeg.
- **Fix**: Install FFmpeg via PowerShell:
  ```powershell
  winget install Gyan.FFmpeg
  ```
  Then restart your terminal. As a temporary workaround, you can convert your audio to `.wav` format.

---

### 5. `Address already in use` or Port Conflict (`8000` or `8501`)
- **Cause**: A previous instance of FastAPI or Streamlit is still running in the background.
- **Fix**:
  - Close any existing terminal windows running Uvicorn or Streamlit.
  - Or kill the process using port 8000 / 8501 in Command Prompt:
    ```cmd
    # Find process ID:
    netstat -ano | findstr :8000
    # Terminate process (replace <PID> with the ID from the previous command):
    taskkill /PID <PID> /F
    ```
  - Or run Uvicorn on a different port:
    ```bash
    uvicorn backend.main:app --reload --port 8001
    ```
    *(Note: If you change the backend port, update `BACKEND_URL` in `frontend/app.py`).*

---

### 6. `ModuleNotFoundError: No module named 'backend'`
- **Cause**: Running the command from inside the `backend/` or `frontend/` directory instead of the project root.
- **Fix**: Ensure your terminal working directory is the root `AudioArtifact` folder:
  ```cmd
  cd /d D:\audioartifact
  uvicorn backend.main:app --reload --port 8000
  ```

---

## 📄 License
Academic and forensic research project. Distributed under the MIT License.