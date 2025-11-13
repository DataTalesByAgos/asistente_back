# Dockerfile for Python FastAPI backend
# Builds a production-ready image with all dependencies

FROM python:3.11-slim
WORKDIR /app

# Install system deps needed for Python packages (especially sentence-transformers, faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy backend source code
COPY backend/ .

# Expose port (Render will use PORT env var or this default)
EXPOSE 8000

# Run FastAPI app with gunicorn + uvicorn worker
# Adjust workers (-w) based on your needs; 1-2 is typical for small apps
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "main:app"]
