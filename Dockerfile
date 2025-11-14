# Build a slim FastAPI image with cached dependencies.
FROM python:3.11-slim AS app

# Keep Python lean and readable logs.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# System deps needed for audio processing (ffmpeg).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Cache dependency install separately.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in the application code.
COPY . .

# Expose the service through Uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]

