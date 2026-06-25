FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; \
               hf_hub_download(repo_id='alihraza/violence-detector', \
                               filename='best_violence_transformer.pth', \
                               local_dir='.')"

RUN chmod +x start.sh

# Streamlit UI + FastAPI metrics/REST
EXPOSE 8501 8000

CMD ["./start.sh"]
