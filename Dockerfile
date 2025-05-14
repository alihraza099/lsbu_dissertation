# Dockerfile for Streamlit app with OpenCV
FROM python:3.10-slim

# Set working directory
WORKDIR /app

#  Install system dependencies needed for OpenCV and Streamlit
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

#  Expose Streamlit's default port
EXPOSE 8501

#  Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]