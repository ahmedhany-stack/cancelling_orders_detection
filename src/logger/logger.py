import os
import sys
import logging
from datetime import datetime

# إنشاء مجلد logs إذا لم يكن موجودًا
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# اسم ملف اللوج
LOG_FILE = os.path.join(LOG_DIR, "log.log")

# إنشاء Logger
logger = logging.getLogger("MLProject")
logger.setLevel(logging.INFO)

# منع تكرار الـ Handlers عند إعادة الاستيراد
if not logger.handlers:

    # الكتابة داخل الملف
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")

    # الطباعة في الـ Terminal
    console_handler = logging.StreamHandler(sys.stdout)

    # شكل اللوج
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)