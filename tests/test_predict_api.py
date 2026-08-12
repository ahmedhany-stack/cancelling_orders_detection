import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# 1. بنستورد الـ Router اللي أنت كتبته فوق
from src.api.predict import router  # أضف المسار الصح للراوتر عندك

# 2. بنعمل تطبيق FastAPI مؤقت للاختبار وبنربط بيه الـ Router
app = FastAPI()
app.include_router(router)

# 3. بنعمل TestClient من مكتبة FastAPI
client = TestClient(app)


# -------------------------------------------------------------
# Test 1: اختبار إن الـ API شغال وبيرجع نتيجة صح مع داتا مظبوطة
# -------------------------------------------------------------
def test_predict_success():
    payload = {
        "invoice": "536365",
        "stock_code": "85123A",
        "description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "quantity": 6,
        "invoice_date": "2026-08-01 08:26:00",
        "price": 2.55,
        "customer_id": "17850",  # 👈 حطينا علامات تنصيص عشان تبقى str زى الـ Schema بالضبط
        "country": "United Kingdom",
    }

    response = client.post("/predict/", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "cancel_probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["cancel_probability"], float)


# -------------------------------------------------------------
# Test 2: اختبار السلوك لما يتبعت داتا ناقصة (Missing Required Field)
# -------------------------------------------------------------
def test_predict_invalid_data():
    # بعتنا payload ناقص خانة إجبارية زي (quantity أو price)
    bad_payload = {
        "invoice": "536365",
        "description": "INVALID ORDER",
    }

    response = client.post("/predict/", json=bad_payload)

    # FastAPI بالتعاون مع Pydantic لازم يرفض الطلب بـ 422
    assert response.status_code == 422