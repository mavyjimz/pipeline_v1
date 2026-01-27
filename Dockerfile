# STAGE 1: The Builder (We do the heavy work here)
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
# We install everything into a specific folder called /install
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# STAGE 2: The Final Image (This is what stays on your D: drive)
FROM python:3.11-slim
WORKDIR /app
# We ONLY grab the finished packages from the builder, leaving the trash behind
COPY --from=builder /install /usr/local
# Copy your actual pipeline scripts
COPY . .

CMD ["python", "src/main_pipeline.py"]