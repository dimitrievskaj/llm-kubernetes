FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip && \
    pip install torch transformers fastapi uvicorn

RUN mkdir /app
WORKDIR /app
COPY run.py .

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]