# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install build tools and CMake
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY pyproject.toml poetry.lock /app/

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --only main

# Copy the rest of the application code to the working directory
COPY . /app

# Copy .env file to the working directory if it exists
RUN if [ -f .env ]; then cp .env /app/.env; fi

# Expose the port that the app will run on
EXPOSE 6000

# Set the command to run the application
CMD ["python", "main.py"]
