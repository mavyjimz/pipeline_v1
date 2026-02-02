FROM python:3.9-slim

# Force absolute paths for the root user
ENV PATH="/usr/local/bin:/usr/bin:/bin:/root/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Install requirements directly to system site-packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files
COPY . .

EXPOSE 8501

# Use the absolute path to the python module
ENTRYPOINT ["python3", "-m", "streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]