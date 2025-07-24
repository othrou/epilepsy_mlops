# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY patient_data_pull.py .

# Default command
CMD ["python", "patient_data_pull.py"]
