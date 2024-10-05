# stages/deployment/app/Dockerfile

# Base image for Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy Streamlit code
COPY . /app

# Install required libraries
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Healthcheck to ensure the container is running
HEALTHCHECK --interval=5m --timeout=3s \
  CMD curl -f http://localhost:${PORT}/ || exit 1

# Set default environment variables for the app
ENV PORT=8501 \
    FASTAPI_URL="http://api:8000"

# Expose the Streamlit port
EXPOSE ${PORT}

# Command to run Streamlit
CMD ["sh", "-c", "streamlit run main.py --server.port=${PORT} --server.enableCORS=false"]