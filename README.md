# 🛒 E-Retail Cancellation Prediction & MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) pipeline for predicting e-commerce order cancellations. This project automates the entire lifecycle—from data preprocessing and model training to containerization, automated CI/CD deployment, and real-time Slack alerting.

---

## 🚀 Project Overview
The goal of this project is to accurately predict whether an online retail order will be cancelled, enabling proactive business interventions. The infrastructure is fully automated using **GitHub Actions**, containerized with **Docker**, tracked via **MLflow & DagsHub**, and monitored through **Slack notifications**.

---

## 🛠️ Tech Stack & Tools
* **Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas, NumPy
* **API Framework:** FastAPI
* **Experiment Tracking & Registry:** MLflow, DagsHub
* **Containerization:** Docker, Docker Compose
* **CI/CD Automation:** GitHub Actions
* **Monitoring & Alerts:** Slack Webhooks
* **Testing:** Pytest

---

## 📂 Repository Structure
```text
cancelling_orders_detection/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yaml         # Automated Build, Push & Slack Alert Pipeline
├── artifacts/
│   ├── models/                # Trained ML models (.pkl / .joblib)
│   └── preprocessors/         # Scalers and encoders
├── src/                       # Source code for training, inference, and pipelines
├── Dockerfile                 # Container configuration for the prediction API
├── requirements.txt           # Project dependencies
├── monitor.py                 # System and pipeline monitoring script
└── README.md                  # Project documentation



MLOps Pipeline & CI/CD Workflow

Whenever code is pushed to the main branch, the GitHub Actions CI/CD pipeline automatically executes the following steps:

    Checkout Code: Retrieves the latest repository state.

    Setup Python & Test: Installs dependencies and runs unit tests via Pytest.

    Docker Buildx: Sets up advanced container building with caching mechanisms.

    Docker Hub Authentication: Securely logs in using encrypted repository secrets.

    Build & Push Image: Compiles the application container and pushes the latest image directly to Docker Hub:

        Repository: ahmed93847/churn-prediction

        Tag: latest

    Slack Alerts: Dispatches instant success/failure status notifications directly to the designated team channel.

🔐 Environment Variables & Secrets

To run this project securely, configure the following secrets in your GitHub Repository (Settings > Secrets and variables > Actions):

    DOCKER_USERNAME: Your Docker Hub username.

    DOCKER_PASSWORD: Your Docker Hub access token/password.

    MLFLOW_TRACKING_URI: DagsHub tracking server URL.

    MLFLOW_TRACKING_USERNAME: DagsHub username.

    MLFLOW_TRACKING_PASSWORD: DagsHub access token.

    SLACK_WEBHOOK_URL: Slack Incoming Webhook URL for build notifications.

🚀 Getting Started Locally

    Clone the repository:
    Bash

    git clone [https://github.com/ahmedhany-stack/cancelling_orders_detection.git](https://github.com/ahmedhany-stack/cancelling_orders_detection.git)
    cd cancelling_orders_detection

    Create and activate a virtual environment:
    Bash

    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\Activate

    Install dependencies:
    Bash

    pip install -r requirements.txt

    Set up your local .env file:
    Create a .env file in the root directory and add your keys:
    مقتطف الرمز

    SLACK_WEBHOOK_URL=your_slack_webhook_here
    MLFLOW_TRACKING_URI=your_mlflow_uri_here


