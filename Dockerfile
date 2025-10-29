# Use an official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (if running an API later)
EXPOSE 8080

# Default command
CMD ["python", "main.py"]
