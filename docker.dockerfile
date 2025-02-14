# Steps to Build and Run the Docker Container

# Build the Docker Image: Open a terminal in the directory containing your Dockerfile and run:
#     docker build -t rag-tariff-calculator .

# Run the Docker Container: After the image is built, you can run the container with:
#     docker run -p 8501:8501 -e GROQ_API_KEY=your_actual_api_key rag-tariff-calculator
# Replace your_actual_api_key with your actual Groq API key.

# Additional Notes
# Poppler Path: Ensure that the path to Poppler is correctly set in your application. 
# The Dockerfile installs Poppler in the default location, so you might need to adjust the 
# path in your code if necessary.
# Environment Variables: The GROQ_API_KEY is set as an environment variable in the Docker container.

#  Make sure to replace "your_api_key_here" with your actual API key or pass it during the docker run command as shown above.
# This Dockerfile sets up a minimal Python environment, installs the necessary dependencies, and runs your Streamlit application. It also installs Poppler for PDF processing, which is required by your application.

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
ENV GROQ_API_KEY="your_api_key_here"

# Run the application
CMD ["streamlit", "run", "app.py"]


