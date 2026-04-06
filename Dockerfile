FROM python:3.11-slim

WORKDIR /app

# gcc is needed to compile some Python C extensions (e.g. aiosqlite wheels)
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# DB and outputs are on a persistent fly.io volume mounted at /data at runtime.
# These ENV vars point the app at the correct paths inside the container.
ENV DB_PATH=/data/job_market.db
ENV OUTPUTS_DIR=/data/outputs

EXPOSE 8080

# At container start:
#   1. Create the persistent volume directories (no-op if already exist).
#   2. Apply the DB schema idempotently (CREATE TABLE IF NOT EXISTS).
#   3. Launch the FastAPI server.
CMD ["sh", "-c", \
     "mkdir -p /data/outputs && \
      python -c 'from db.connection import init_db; init_db()' && \
      uvicorn api.main:app --host 0.0.0.0 --port 8080"]
