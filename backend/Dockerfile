FROM python:3.10-slim

WORKDIR /app

# 1. ลง Dependencies พื้นฐานของ Linux ที่จำเป็น
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 2. ลง Python Libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 🔥 ลง Playwright และ Browser (หัวใจสำคัญ)
# - install-deps: ลงพวก library ระบบที่ chromium ต้องใช้ (เช่น libgtk, libasound)
# - install chromium: ลงเฉพาะ chromium เพื่อประหยัดที่ (ไม่ต้องลง firefox/webkit)
RUN playwright install-deps
RUN playwright install chromium

COPY . .

# 4. สร้างโฟลเดอร์เก็บข้อมูล (กัน Error เรื่อง Permission)
RUN mkdir -p /app/data/processed && chmod -R 777 /app/data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]