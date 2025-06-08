# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 5000 (Flask default)
EXPOSE 5000

# Command to run the application
CMD ["python", "test.py"]

# Note: This Dockerfile assumes that you have a .env file with the following variables:
# COSMOS_DB_URI
# COSMOS_DB_KEY
# DATABASE_NAME
# CONTAINER_NAME
# AZURE_OPENAI_API_KEY
# AZURE_OPENAI_ENDPOINT
# AZURE_OPENAI_EMBEDDING_DEPLOYMENT
# AZURE_OPENAI_GPT4O_DEPLOYMENT
# AZURE_OPENAI_API_VERSION
