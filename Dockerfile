# Use the official lightweight Python image based on Debian
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libhdf5-dev \
    cmake \
    git \
    && apt-get clean

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV FLASK_APP=main.py
# Optionally set the FLASK_ENV to development for debug mode
# ENV FLASK_ENV=development

# Expose the port the app runs on
EXPOSE 5000

# Specify the command to run the application
CMD ["flask", "run", "--host", "0.0.0.0"]
