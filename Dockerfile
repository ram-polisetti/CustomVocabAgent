FROM python:3.12-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Create a smaller final image
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY . /app

EXPOSE 80

# Set environment variables (if needed)
# ENV GROQ_API_KEY=your_groq_api_key_here

# Command to run the application
CMD ["python", "main.py"]