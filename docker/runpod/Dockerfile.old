# Sillion Valiey Standard: RunPod Serverless Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Ensure 'python' command points to 'python3'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git curl ffmpeg libsm6 libxext6 libgl1-mesa-glx build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for the application
WORKDIR /app

# Copy application files into the container
COPY src/ /app/

# Install Python dependencies from requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch packages explicitly with CUDA 11.8 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set the container entrypoint to RunPod's handler script
ENTRYPOINT ["python", "-u", "handler.py"]