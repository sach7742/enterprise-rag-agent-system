#!/usr/bin/env bash
set -e

# Set project root and src directory in PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/src

cleanup() {
    echo ""
    echo "Stopping Enterprise RAG Agent System..."
    kill $(jobs -p) 2>/dev/null || true
    echo "Done."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting Enterprise RAG Agent System..."
echo "---------------------------------------"

# 1. Start FastAPI backend
echo "Launching FastAPI Backend on http://127.0.0.1:8000..."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &

# Pause for backend initialization
sleep 2

# 2. Start Streamlit frontend
echo "Launching Streamlit Frontend on http://127.0.0.1:8501..."
streamlit run app_ui.py --server.port 8501 &

echo "---------------------------------------"
echo "System active! Press Ctrl+C to terminate both services."
wait
