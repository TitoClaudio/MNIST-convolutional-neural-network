FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY src/ ./src/
COPY config.py ./
COPY trained_cnn_model.npz ./

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
