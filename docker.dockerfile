# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Poppler for PDF processing
RUN apt-get update && apt-get install -y poppler-utils

# Expose the port the app runs on
EXPOSE 8501

# Define environment variable for the Groq API key
ENV GROQ_API_KEY="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"

# Run the application
CMD ["streamlit", "run", "app.py"]