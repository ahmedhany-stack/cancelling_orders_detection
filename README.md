🛒 E-Commerce Cancellation & Churn Detection System

[![CI/CD Pipeline](https://github.com/ahmedhany-stack/cancelling_orders_detection/actions/workflows/main.yml/badge.svg)](https://github.com/ahmedhany-stack/cancelling_orders_detection/actions)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Pytest](https://img.shields.io/badge/Tests-Pytest%20Passed-success?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end production-ready MLOps system that predicts customer order cancellations and churn probabilities in real-time. Built with **FastAPI**, containerized via **Docker**, and validated using continuous integration (**GitHub Actions & Pytest**).

---

## 📐 System Architecture

```text
[ Client / HTTP Request ]
           │
           ▼
  ┌─────────────────┐
  │   FastAPI API   │ ──( Pydantic Validation & Schema Check )
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  CustomData     │ ──( Data Preprocessing & DataFrame Structuring )
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Inference Pipeline │ ──( Feature Engineering & ML Model Scoring )
  └────────┬────────┘
           │
           ▼
[ JSON Prediction Output ] ──( prediction status + cancel_probability )

✨ Key Features

    High-Performance API: Powered by FastAPI with strictly enforced Pydantic request/response validation schemas.

    Modular Pipeline: Decoupled data parsing (CustomData) and inference engine (PredictionPipeline).

    Automated Testing: 100% test coverage for API endpoints (happy path & 422 edge cases) and data transformation pipelines using Pytest.

    Containerized Deployment: Packaged into lightweight Docker containers ready for any cloud orchestrator (AWS EC2, Render, Kubernetes).

    CI/CD Integration: Automated GitHub Actions workflows executing full test suites on every push and pull_request.

🛠️ Tech Stack

    Language: Python 3.10

    API Framework: FastAPI, Uvicorn, Pydantic

    Machine Learning & Data Processing: Pandas, NumPy, Scikit-learn, Joblib

    Testing & Quality Assurance: Pytest, HTTPX, TestClient

    DevOps & MLOps: Docker, GitHub Actions, UV / Pipenv

🚀 Quick Start
Option 1: Run Locally with Python

    Clone the Repository:
    Bash

    git clone [https://github.com/ahmedhany-stack/cancelling_orders_detection.git](https://github.com/ahmedhany-stack/cancelling_orders_detection.git)
    cd cancelling_orders_detection

    Set up Virtual Environment:
    Bash

    python -m venv .venv
    # Activate Environment:
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    # Linux/macOS: source .venv/bin/activate

    Install Dependencies:
    Bash

    pip install -r requirements.txt

    Run FastAPI Server:
    Bash

    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

    Access Interactive API Docs (Swagger UI) at: http://localhost:8000/docs

Option 2: Run via Docker Container

    Build the Docker Image:
    Bash

    docker build -t sales-cancellation-api .

    Run the Container:
    Bash

    docker run -p 8000:8000 sales-cancellation-api

🧪 Running Unit & Integration Tests

The test suite validates data transformation schemas, API behavior, and error handling.

To run tests locally:
Bash

python -m pytest tests/ -v

Expected Output:
Plaintext

tests/test_pipeline.py::test_custom_data_to_dataframe PASSED      [ 25%]
tests/test_pipeline.py::test_prediction_pipeline_output PASSED   [ 50%]
tests/test_predict_api.py::test_predict_success PASSED          [ 75%]
tests/test_predict_api.py::test_predict_invalid_data PASSED     [100%]

========================== 4 passed in 3.61s ==========================

📡 API Endpoint Reference
POST /predict/

Performs order cancellation prediction for a given e-commerce transaction.
Request Body Sample:
JSON

{
  "invoice": "536365",
  "stock_code": "85123A",
  "description": "WHITE HANGING HEART T-LIGHT HOLDER",
  "quantity": 6,
  "invoice_date": "2026-08-01 08:26:00",
  "price": 2.55,
  "customer_id": "17850",
  "country": "United Kingdom"
}

Response Body Sample (200 OK):
JSON

{
  "prediction": 1,
  "cancel_probability": 0.875
}

Error Response (422 Unprocessable Entity):

Returned automatically when required schema attributes are missing or misformatted.
📁 Repository Structure
Plaintext

cancelling_orders_detection/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD pipelines
│       └── main.yml
├── src/
│   ├── api/                # FastAPI routes & Pydantic schemas
│   │   ├── main.py
│   │   └── schemas.py
│   ├── pipeline/           # Inference pipeline & Data transformers
│   │   └── prediction_pipeline.py
│   └── components/         # Preprocessing and model training code
├── tests/                  # Pytest test suite
│   ├── test_predict_api.py
│   └── test_pipeline.py
├── Dockerfile              # Docker container configuration
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

👤 Author

    Ahmed Hany - Machine Learning & MLOps Engineer

    GitHub: @ahmedhany-stack

Developed with focus on Scalability, Maintainability, and Production MLOps Best Practices.


---

### 💡 نصيحة قبل الـ Push:
تأكد فقط أن مسار رابط الـ Badge في السطر الأول يطابق اسم ملف الـ workflow عندك في GitHub (لو ملف الـ workflow اسمه `main.yml` يبقى الرابط مظبوط 100%).

اعمل `git add README.md` و `git commit -m "Add professional README documentation"` و `git push` ليكون وجهة متكاملة على الـ Repository! 😎🔥