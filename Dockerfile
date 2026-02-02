FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:/usr/bin:/bin:/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install with --force-reinstall to ensure no ghost links
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# DISCOVERY: This will print exactly where streamlit is during the build
RUN which streamlit && streamlit --version

EXPOSE 8501
ENTRYPOINT ["python3", "-m", "streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]