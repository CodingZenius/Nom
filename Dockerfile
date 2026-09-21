FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    NOMAD_HOME=/data/nomad \
    NOMAD_WORKSPACE=/workspace

# curl/zstd: Ollama download+extract. procps: process tools. Chromium libs come from playwright install-deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates zstd git procps build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY . .
VOLUME ["/data", "/workspace"]
# Ollama, the model and Chromium are downloaded on first boot into /data (once, then cached).
ENTRYPOINT ["python", "run.py"]
CMD ["chat"]
