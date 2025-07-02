FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your project files
COPY . .

# Expose the port
EXPOSE 10000

# Run with gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:10000", "app:app"]
