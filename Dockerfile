# Use a base image with GPU support
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI and run app
EXPOSE 8000
CMD ["uvicorn", "telegrambot:app", "--host", "0.0.0.0", "--port", "8000"]
