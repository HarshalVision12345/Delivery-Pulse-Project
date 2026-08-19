#!/bin/sh

# Start FastAPI backend in the background
uvicorn delivery_api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend in the foreground
streamlit run app.py --server.$port=8501 --server.address=0.0.0.0