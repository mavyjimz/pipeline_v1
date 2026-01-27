# STAGE 1: Builder (The "Kitchen")
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STAGE 2: Runner (The "Lean Machine")
FROM python:3.9-slim

WORKDIR /app

# Copy only the installed packages from the builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your source code
COPY . .

# Lesson #41: Heartbeat (Healthcheck)
# This checks if the main script exists as a proxy for a healthy container
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import os; exit(0 if os.path.exists('/app/src/main_pipeline.py') else 1)"

# Execute the pipeline
CMD ["python", "src/main_pipeline.py"]