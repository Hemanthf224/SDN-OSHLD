FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Exclude the windows specific torch wheel
RUN grep -v "https://download.pytorch.org/whl/cu121" requirements.txt > req_linux.txt
RUN pip install --no-cache-dir -r req_linux.txt
# Install Linux PyTorch CPU for docker since typical docker doesn't expose GPU without nvidia-docker
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

# Environment variables
ENV PYTHONPATH=/app
