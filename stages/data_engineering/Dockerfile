# stages/data_engineering/Dockerfile

# Base image for Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy FastAPI code
COPY . /app

# Install required libraries
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set default environment variables for the app
ENV PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=4

# Healthcheck to ensure the container is running
HEALTHCHECK --interval=5m --timeout=3s \
  CMD curl -f http://localhost:${PORT}/ || exit 1

# Expose the port FastAPI will run on
EXPOSE ${PORT}

# Command to run FastAPI
CMD ["sh", "-c", "uvicorn main:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]
