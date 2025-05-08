FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Detectron2 (pastikan torch sudah terinstall)
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy app code
COPY . .

# Expose port
EXPOSE 5000

# Command to run app
CMD ["python", "app.py"]
