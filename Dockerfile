# Use an official Python 3.10 base image
#FROM pytorch/pytorch:2.7.0-cuda12.5-cudnn8-runtime
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install dependencies
# We'll use --no-cache-dir to reduce image size
# Some dependencies might need build tools, so let's install them first
# and clean them up afterwards if possible (though for -slim it might be tricky
# without adding too much complexity for now).
# swig is listed in requirements and often needs to be installed via apt
RUN apt-get update && \
    apt-get install -y swig build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove swig build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the project files into the working directory
COPY . .

# Users can run scripts using:
# docker run <image_name> python train_world_model_parallel.py