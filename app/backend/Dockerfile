FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/refineshot-requirements.txt
COPY app/backend/requirements.txt /tmp/web-requirements.txt
RUN pip install --no-cache-dir \
        -r /tmp/refineshot-requirements.txt \
        -r /tmp/web-requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY app/backend ./web/backend

WORKDIR /app/web/backend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
