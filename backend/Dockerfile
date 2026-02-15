FROM python:3.10-slim

WORKDIR /app

# 1. Install required basic Linux dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Install Playwright and Browser (Core for web scraping)
# - install-deps: Install system libraries required by chromium (e.g., libgtk, libasound)
# - install chromium: Install only chromium to save disk space (skip firefox/webkit)
RUN playwright install-deps
RUN playwright install chromium

COPY . .

# 4. Create data storage directory (Prevent permission errors)
RUN mkdir -p /app/data/processed && chmod -R 777 /app/data

# 5. Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]