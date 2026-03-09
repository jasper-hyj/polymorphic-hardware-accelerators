FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────
# Build iverilog from latest source for better SystemVerilog support
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        autoconf \
        bison \
        flex \
    gperf \
    && git clone --depth 1 https://github.com/steveicarus/iverilog.git /tmp/iverilog && \
    cd /tmp/iverilog && \
    autoconf && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/iverilog && \
    apt-get remove -y git autoconf bison flex gperf && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────
COPY run.py .
COPY config.yaml .
COPY src/ src/
COPY demo_projects/ demo_projects/

# ── Volumes expected at runtime ───────────────────────────────────────
#   /app/verilog_proj  — mount your RTL projects here
#   /app/output        — reports and figures appear here
VOLUME ["/app/verilog_proj", "/app/output"]

ENTRYPOINT ["python", "run.py"]
CMD []
