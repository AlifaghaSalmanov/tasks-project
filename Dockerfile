# Build a slim FastAPI image with cached dependencies.
FROM python:3.11-slim AS app

# Keep Python lean and readable logs.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Cache dependency install separately.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in the application code.
COPY . .

# Expose the service through Uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000", "--ssl-keyfile", "cert/key.pem", "--ssl-certfile", "cert/cert.pem"]

