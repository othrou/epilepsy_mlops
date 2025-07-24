# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy evaluate script
COPY evaluate.py .

# Default command to run the evaluate script
CMD ["python", "evaluate.py"]
