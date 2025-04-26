# Use a lightweight official Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for wget and unzip
# clean up apt lists afterwards to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# Use --no-cache-dir to prevent pip from caching wheels, reducing image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# The GloVe download/load logic is handled by app.py when Gunicorn starts it.
# We need to ensure the 'glove' directory exists and is where app.py expects it.
# The docker-compose file will use a volume for /app/glove, so we don't need
# to pre-download here, but we should create the directory structure.
RUN mkdir -p glove

# Expose the port the app will run on
EXPOSE 4001

# Command to run the application using Gunicorn
# -w: number of worker processes (e.g., 4)
# -b: bind to host and port
# app:app: The Flask application instance 'app' in the 'app.py' file
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:4001", "app:app"]