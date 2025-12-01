# For more information, please refer to https://aka.ms/vscode-docker-python
# Use a slim Python 3.11 base image for smaller size
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# --- Dependency Installation Phase (Leverages Caching) ---
# 1. Copy only the requirements file first.
#    This layer is only rebuilt if requirements.txt changes.
COPY requirements.txt .

# 2. Install OS dependencies needed by Python packages
#    (We'll keep the essential ones for common ML libraries just in case)
#    Added python3-dev for building some wheels.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Dependencies for building some Python packages (e.g. Scipy, Pandas)
        gcc \
        python3-dev \
        # Dependencies for 'curl' is often needed for fetching data
        curl \
        && rm -rf /var/lib/apt/lists/*

# 3. Install Python dependencies
#    --no-cache-dir reduces image size.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Application Copy Phase (Last step to maintain cache) ---
# 4. Copy the rest of the application code
COPY . .
    
# Port the application listens on (as defined in your system architecture)
EXPOSE 5000

# Set environment variables for Flask (replace app.py with your main Flask app file)
ENV FLASK_APP=app.py

# Default run command: Use -u for unbuffered output (better for logging)
CMD ["python", "-u", "app.py"]

