FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies (IMPORTANT)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "serving.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
