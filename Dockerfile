FROM python:3.13-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
COPY .streamlit/ ./.streamlit/

# Expose API and dashboard ports
EXPOSE 8000 8501

