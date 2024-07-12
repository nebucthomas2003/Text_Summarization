FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first for caching
COPY requirements.txt /app/

# Install dependencies with retry and increased timeout
RUN pip install --no-cache-dir --default-timeout=100 torch==1.9.0

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Copy the saved model files
COPY saved_model /app/saved_model

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]
