FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# تثبيت الأدوات الأساسية للنظام
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uvicorn fastapi && \
    pip install --no-cache-dir -r requirements.txt

# نسخ كود المشروع والموديلات الجاهزة بالكامل
COPY . .

EXPOSE 8000

# تشغيل الـ API مباشرة فور قيام الـ Container
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]