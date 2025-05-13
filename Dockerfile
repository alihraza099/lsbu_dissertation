# Use official Python base image
FROM python:3.10-slim

# Install required OS packages
COPY packages.txt .
RUN apt-get update && \
    xargs -a packages.txt apt-get install -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
