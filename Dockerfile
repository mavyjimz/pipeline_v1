# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create the internal bridge folders (Matching your Compose mounts)
RUN mkdir -p /app/input_data /app/shared_output

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the exact path to the execution script
ENTRYPOINT ["python3", "-m", "streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]