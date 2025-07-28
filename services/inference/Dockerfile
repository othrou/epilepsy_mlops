# services/inference/Dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inference_api.py .

CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8001"]
