FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    PATH="/root/.local/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN curl -fsSL https://astral.sh/uv/install.sh | sh

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --upgrade --link-mode=copy -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]