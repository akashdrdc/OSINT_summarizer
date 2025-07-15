# Use official Python slim image
FROM python:3.9-slim

# Set working dir
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Launch Streamlit on $PORT (Cloud Run sets this)
CMD ["streamlit", "run", "app.py", \
     "--server.port", "8080", \
     "--server.address", "0.0.0.0", \
     "--server.enableCORS", "false"]
