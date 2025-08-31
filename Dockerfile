FROM python:3.11-slim

WORKDIR /app

# Needed for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8008

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8008"]
