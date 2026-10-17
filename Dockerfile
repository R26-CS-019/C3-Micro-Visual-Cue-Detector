# C3 Micro Visual Cue API (TensorFlow CNNs + headless Chromium) for Azure Container Apps.
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    C3_HEADLESS=1

ARG C3_BUILD_REVISION=unknown
ENV C3_BUILD_REVISION=${C3_BUILD_REVISION} C3_CAPTURE_WAIT_MS=7000

WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

WORKDIR "/app/Micro Visual Cue Component"
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "integration.visual_api:app", "--host", "0.0.0.0", "--port", "8000"]
