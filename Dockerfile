FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY .env .

# Set environment variables (if needed)
# ENV GROQ_API_KEY=your_groq_api_key_here

# Command to run the application
CMD ["python", "main.py"]
