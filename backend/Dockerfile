FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies (including build tools for llama-cpp-python)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Install Playwright and Browser (Core for web scraping)
RUN pip install --no-cache-dir playwright
RUN playwright install chromium

# 4. Copy application files
COPY . .

# 5. Expose port
EXPOSE 8000

# 6. Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]