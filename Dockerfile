# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /code

# ── Install Python dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY app/ ./app/

# ── HuggingFace Spaces runs as a non-root user ─────────────────────────────────
# Create a non-root user and switch to it
RUN useradd -m -u 1000 mitosys
USER mitosys

# ── Expose HuggingFace Spaces default port ─────────────────────────────────────
EXPOSE 7860

# ── Start the server ───────────────────────────────────────────────────────────
# HF Spaces expects the app on port 7860
# OPENAI_API_KEY is injected by HF as an environment variable via Secrets
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]