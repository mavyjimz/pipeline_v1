# Use a specific, stable version of Python
FROM python:3.9-slim

# Set environment variables to force the PATH
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for safety
RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser/app

# Copy and install requirements with extra patience
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --user --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the rest of the code
COPY --chown=appuser:appuser . .

# Expose the Streamlit port
EXPOSE 8501

# The "Bulletproof" launch command
ENTRYPOINT ["python3", "-m", "streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]