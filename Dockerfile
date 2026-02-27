# Multi-stage Dockerfile for MindScribe

# Stage 1: Base image with Python
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application image
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files from base stage
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
