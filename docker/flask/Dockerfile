# Sillion Valiey Standard: Flask Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Ensure 'python' command points to 'python3'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    cmake && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory and create application directory
WORKDIR /app
RUN mkdir -p /app/CatVTON
WORKDIR /app/CatVTON

# Copy application files
COPY src/ /app/CatVTON/

# Install Python dependencies including PyTorch with CUDA support
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional Python packages for Flask
RUN pip install --no-cache-dir flask boto3 requests

# Expose Flask port
EXPOSE 5000

# Run the Flask application
CMD ["python", "api_app.py"]