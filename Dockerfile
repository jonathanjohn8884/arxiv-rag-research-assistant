FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for FAISS and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

# Runtime directories
RUN mkdir -p vector_stores/faiss_index results data/sample_papers

EXPOSE 7860

CMD ["python", "api/main.py"]
