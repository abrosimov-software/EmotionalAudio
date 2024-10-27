# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . .

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Test if GPU is accessible (Optional, remove if unnecessary)
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"