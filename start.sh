#!/bin/bash

echo "Starting PDF Converter Pro with Authentication..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Start PostgreSQL database
echo "Starting PostgreSQL database..."
docker compose up -d postgres

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 5

# Check if database is running
if ! docker compose ps postgres | grep -q "Up"; then
    echo "Error: Failed to start PostgreSQL database."
    exit 1
fi

echo "PostgreSQL database is running!"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start the FastAPI application
echo "Starting FastAPI application..."
python backend.py

echo "Application started! Visit http://localhost:8000 to access the login page." 