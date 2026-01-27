# 1. Use a lightweight Python 3.10 foundation
FROM python:3.10-slim

# 2. Set the internal factory floor to /app
WORKDIR /app

# 3. Copy only the shopping list first (helps with speed)
COPY requirements.txt .

# 4. Install the tools in CPU mode
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all our Lesson 35 scripts into the container
COPY . .

# 6. Run the pipeline when the container starts
CMD ["python", "src/main_pipeline.py"]