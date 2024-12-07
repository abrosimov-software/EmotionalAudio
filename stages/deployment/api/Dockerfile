# stages/deployment/api/Dockerfile

# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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

# Expose the port FastAPI will run on
EXPOSE ${PORT}

# Command to run FastAPI
CMD ["sh", "-c", "uvicorn main:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]
