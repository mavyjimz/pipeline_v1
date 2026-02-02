FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# We add every possible path we've suspected today
ENV PATH="/usr/local/bin:/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Using the most standard install possible
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Instead of running the dashboard, we tell the container to just wait for us
ENTRYPOINT ["tail", "-f", "/dev/null"]