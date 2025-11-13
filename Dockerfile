# Dockerfile for Python FastAPI backend
# Builds a production-ready image with all dependencies

FROM python:3.11-slim
WORKDIR /app

# Install system deps needed for Python packages (especially sentence-transformers, faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy backend source code (assumes Docker build context is the `backend/` folder)
COPY . .

# Expose port (Render will set $PORT for you)
EXPOSE 8000

# Use shell form so ${PORT} is expanded at runtime; default to 8000 if not set
CMD ["sh", "-c", "gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000} main:app"]
