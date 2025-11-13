# Dockerfile for Python FastAPI backend
# Builds a production-ready image with all dependencies

FROM python:3.11-slim
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code (assumes Docker build context is the `backend/` folder)
COPY . .

# Expose port (Render will set $PORT for you)
EXPOSE 8000

# Use uvicorn directly (lighter) and expand ${PORT} at runtime
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
