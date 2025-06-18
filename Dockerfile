# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache

# Create working directory
WORKDIR /app

# --- Install Tesseract and dependencies ---
RUN apt-get update && \
    apt-get install -y \
        tesseract-ocr \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        poppler-utils \
        && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy app code
COPY . .

# Ensure HuggingFace cache is writable
RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

# Expose port
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
