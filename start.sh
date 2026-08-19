#!/bin/sh

# Start FastAPI backend internally on port 8000
uvicorn delivery_api:app --host 127.0.0.1 --port 8000 &

# Start Streamlit on the external platform $PORT
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0