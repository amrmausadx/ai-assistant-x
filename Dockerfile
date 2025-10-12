# For more information, please refer to https://aka.ms/vscode-docker-python
# Use a slim Python 3.11 base image for smaller size
FROM python:3.11-slim

WORKDIR /app
COPY . /app



# Install OS dependencies needed by Python packages
#--no-install-recommends reduces unnecessary packages.
#rm -rf /var/lib/apt/lists/* cleans up cache to reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        gfortran \
        libatlas-base-dev \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .
    
    
EXPOSE 6000
# Set environment variables for Flask (replace app.py with your main Flask app file)
ENV FLASK_APP=app.py
# default run (expects you trained locally or included models in the image)
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6000"]
CMD ["gunicorn", "--bind", "0.0.0.0:6000", "--workers", "4", "app:app"]
