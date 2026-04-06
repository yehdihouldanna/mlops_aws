FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies (IMPORTANT)
RUN apk add --no-cache \
    build-base \
    gcc \
    g++ \
    python3-dev \
    musl-dev \
    lapack-dev

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "serving.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
