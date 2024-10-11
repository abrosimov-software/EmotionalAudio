# stages/model_engineering/Dockerfile

# Base image for Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy FastAPI code
COPY . /app

# Install required libraries
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Healthcheck to ensure the container is running
HEALTHCHECK --interval=5m --timeout=3s \
  CMD curl -f http://localhost:${PORT}/ || exit 1

# Set default environment variables for the app
ENV PORT=8000 \
    HOST=0.0.0.0

# Expose the port for MLflow
EXPOSE ${PORT}

# Command to run MLflow
CMD ["sh", "-c", "mlflow server --host ${HOST} --port ${PORT} --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlruns"]
