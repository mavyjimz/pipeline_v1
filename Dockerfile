# Use the official Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for stability
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:/usr/bin:/bin:/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy all project files
COPY . .

# Discovery check to ensure Streamlit is visible
RUN which streamlit && streamlit --version

# Launch the dashboard
ENTRYPOINT ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]