# IDS ML - Dockerfile
# Système de Détection d'Intrusions avec Machine Learning
# Développé par: Rana Romdhane & Oulimata Sall

FROM python:3.11-slim

LABEL maintainer="Rana Romdhane & Oulimata Sall"
LABEL description="Intelligent Intrusion Detection System with Machine Learning"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpcap-dev \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models logs uploads

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/stats')"

# Run the application
CMD ["python", "app.py"]