FROM python:3.10-slim

WORKDIR /app

COPY MLProject /app/MLProject

RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    scikit-learn \
    joblib

CMD ["python", "MLProject/modelling.py"]