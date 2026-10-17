# C3 Micro Visual Cue API (TensorFlow CNNs + headless Chromium) for Azure Container Apps.
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    C3_HEADLESS=1

ARG C3_BUILD_REVISION=unknown
# Capture profile c3-capture-v2 (same as the local interface): full Chromium,
# desktop Chrome identity, en-US, Asia/Colombo, 10-second render wait.
ENV C3_BUILD_REVISION=${C3_BUILD_REVISION} \
    C3_CAPTURE_WAIT_MS=10000 \
    C3_BROWSER_CHANNEL=chromium \
    C3_LOCALE=en-US \
    C3_TIMEZONE=Asia/Colombo

WORKDIR /app
COPY requirements-lock.txt .
# fonts-noto-core adds Sinhala, Tamil and other scripts so Sri Lankan pages
# render with real glyphs, as they do on Windows.
RUN pip install --no-cache-dir -r requirements-lock.txt \
    && python -m playwright install --with-deps chromium \
    && apt-get install -y --no-install-recommends fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

WORKDIR "/app/Micro Visual Cue Component"
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "integration.visual_api:app", "--host", "0.0.0.0", "--port", "8000"]
