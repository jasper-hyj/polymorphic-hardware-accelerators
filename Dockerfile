FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        iverilog \
        git \
        curl \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────
COPY run.py .
COPY src/ src/

# ── Volumes expected at runtime ───────────────────────────────────────
#   /app/verilog_proj  — mount your RTL projects here
#   /app/output        — reports and figures appear here
VOLUME ["/app/verilog_proj", "/app/output"]

ENTRYPOINT ["python", "run.py"]
CMD []
