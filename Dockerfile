FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/data /app/models

ENV MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    LOG_LEVEL=INFO

 # Train model at build time so it's baked into the image  --because of free tier render
RUN python train.py

RUN useradd -m appuser && chown -R appuser:appuser /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["/app/start.sh"]