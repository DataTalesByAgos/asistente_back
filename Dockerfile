# Example Dockerfile for Python backend. Adjust CMD depending on framework.
# Assumes there's a `requirements.txt` at repo root and the app exposes an ASGI/WSGI app
# If your `main.py` exposes a FastAPI/Flask `app` object, this uses gunicorn+uvicorn worker.

FROM python:3.11-slim
WORKDIR /app

# Install system deps needed for many Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app sources
COPY . .

# Default port (Render uses PORT env var)
ENV PORT=8000

# CMD: try to run as ASGI app with gunicorn+uvicorn worker (FastAPI example)
# If your app is plain script, replace with: CMD ["python", "main.py"]
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
