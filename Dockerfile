FROM python:3.9-slim

# Force the PATH to include common pip locations
ENV PATH="/usr/local/bin:/usr/bin:/bin:/root/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy project files
COPY . .

EXPOSE 8501

# DIRECT ENGINE EXECUTION: This bypasses "command not found" errors
ENTRYPOINT ["python3", "-m", "streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]