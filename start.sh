#!/bin/sh
# Start FastAPI (metrics + REST API) in background, then Streamlit in foreground.
# Streamlit exit = container exit.
uvicorn api:app --host 0.0.0.0 --port 8000 &
exec streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.enableCORS=false
