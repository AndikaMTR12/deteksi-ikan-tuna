FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies early to leverage cache
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install detectron2 after torch
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy app files
COPY . .

# Run the app
CMD ["python", "app.py"]
