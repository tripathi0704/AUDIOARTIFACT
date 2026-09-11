#!/usr/bin/env bash
# ==============================================================================
# AudioArtifact v2.0 - VPS Automated Production Deployment Script
# Supports: Ubuntu 20.04 / 22.04 / 24.04, Debian 11 / 12
# ==============================================================================

set -e

echo "=================================================================="
echo "          AudioArtifact v2.0 - VPS Deployment Installer           "
echo "=================================================================="

# 1. Update system & install essential system dependencies
echo "[1/6] Installing system packages (Python, FFmpeg, Nginx, libsndfile)..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv ffmpeg libsndfile1 git nginx curl

# 2. Setup project directory
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# 3. Create Python Virtual Environment
echo "[2/6] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 4. Install Python dependencies
echo "[3/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Setup Systemd Service: Backend (FastAPI on port 8000)
echo "[4/6] Configuring Backend Systemd Service..."
sudo bash -c "cat <<EOF > /etc/systemd/system/audioartifact-backend.service
[Unit]
Description=AudioArtifact FastAPI Backend
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
Environment=\"PATH=$APP_DIR/venv/bin:/usr/local/bin:/usr/bin\"
Environment=\"HF_HUB_OFFLINE=1\"
Environment=\"TRANSFORMERS_OFFLINE=1\"
ExecStart=$APP_DIR/venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"

# 6. Setup Systemd Service: Frontend (Streamlit on port 8501)
echo "[5/6] Configuring Frontend Systemd Service..."
sudo bash -c "cat <<EOF > /etc/systemd/system/audioartifact-frontend.service
[Unit]
Description=AudioArtifact Streamlit Frontend
After=network.target audioartifact-backend.service

[Service]
User=$USER
WorkingDirectory=$APP_DIR
Environment=\"PATH=$APP_DIR/venv/bin:/usr/local/bin:/usr/bin\"
Environment=\"BACKEND_URL=http://127.0.0.1:8000\"
ExecStart=$APP_DIR/venv/bin/streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"

# 7. Configure Nginx Reverse Proxy
echo "[6/6] Configuring Nginx reverse proxy..."
sudo bash -c "cat <<EOF > /etc/nginx/sites-available/audioartifact
server {
    listen 80;
    server_name _;

    client_max_body_size 100M;

    # Streamlit Frontend
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }

    # Streamlit Websocket
    location /_stcore/stream {
        proxy_pass http://127.0.0.1:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header Host \$host;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_read_timeout 86400;
    }

    # FastAPI Backend Docs & direct API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 300;
    }
}
EOF"

sudo ln -sf /etc/nginx/sites-available/audioartifact /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# Reload and start systemd services
echo "Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable audioartifact-backend audioartifact-frontend
sudo systemctl restart audioartifact-backend audioartifact-frontend

PUBLIC_IP=\$(curl -s https://api.ipify.org || echo "YOUR_SERVER_IP")

echo ""
echo "=================================================================="
echo " 🎉 AudioArtifact v2.0 is now LIVE on the Internet!"
echo " 👉 Visit in your browser: http://\$PUBLIC_IP"
echo "=================================================================="
