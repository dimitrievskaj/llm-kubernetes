FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir onnxruntime transformers fastapi uvicorn

COPY run.py .
COPY model ./model

EXPOSE 8000
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]