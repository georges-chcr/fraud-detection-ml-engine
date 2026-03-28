FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Installation des dépendances en premier pour profiter du cache Docker
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copie du code applicatif et des artefacts ML
COPY api.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
