🛒 E-Commerce Order Cancellation Prediction & MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) project designed to predict e-commerce order cancellations. This project automates the full ML lifecycle—including data versioning, modular pipelines, experiment tracking, continuous retraining via CI/CD, drift monitoring, and REST API deployment.
🏗️ Architecture & Technical Workflow

┌─────────────────┐      ┌─────────────────┐      ┌──────────────────┐
│  Data Source    │ ───► │ Data Ingestion  │ ───► │  Transformation  │
│  & DVC Tracking │      │ & Validation    │      │  & Feature Eng.  │
└─────────────────┘      └─────────────────┘      └────────┬─────────┘
                                                           │
                                                           ▼
┌─────────────────┐      ┌─────────────────┐      ┌──────────────────┐
│ Docker Deployment│ ◄─── │ MLflow Tracking │ ◄─── │  Model Trainer   │
│ & FastAPI Serving│      │ & DagsHub Reg.  │      │  (XGBoost / ML)  │
└─────────────────┘      └─────────────────┘      └──────────────────┘

🛠️ Tech Stack & Tools

    Core Programming: Python 3.10

    Data & Machine Learning: Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn

    Data Version Control (DVC): Remote storage synchronization with DagsHub

    Experiment Tracking & Model Registry: MLflow hosted on DagsHub

    API & Serving: FastAPI, Uvicorn, Pydantic

    Containerization: Docker, Docker Compose

    CI/CD & Automation: GitHub Actions (Automated Retraining, Docker Build & Push)

    Model & Data Monitoring: Evidently AI (Data Drift Detection)

    Testing & Quality: Pytest, Flake8, Black

📁 Project Structure
Plaintext

sales_Project/
├── .dvc/                        # DVC configuration directory
├── .github/
│   └── workflows/              # CI/CD pipelines (Retraining, Docker build)
├── artifacts/                   # Local pipeline artifacts (Features, Models)
├── configs/                     # Configuration YAML files
├── data/                        # Raw & Processed datasets
├── logs/                        # Application runtime execution logs
├── notebooks/                   # Exploratory Data Analysis (EDA) & Research
├── reports/                     # Data drift and monitoring reports
├── saved_models/                # Production serialized model binaries
├── saved_preprocessors/         # Scalers & Encoders transformers
├── src/                         # Modular Source Code
│   ├── components/              # Ingestion, Transformation, Trainer pipelines
│   ├── pipeline/                # Prediction & Training orchestration
│   ├── utils/                   # Helper functions & I/O handlers
│   ├── exception.py             # Custom Exception Handler
│   └── logger.py                # Centralized Logging system
├── tests/                       # Unit and Integration Pytest suite
├── app.py                       # FastAPI application entry point
├── monitor.py                   # Data drift detection script (Evidently)
├── Dockerfile                   # Docker container configuration
├── docker-compose.yml           # Container deployment compose file
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup script

🚀 Key MLOps Features Implemented
1. Data Versioning (DVC + DagsHub)

All datasets (.csv, .xlsx) are tracked outside of Git using DVC. Large feature sets are pushed to a remote DagsHub storage bucket, maintaining lightweight Git repository commits while guaranteeing data reproducibility.
2. Experiment Tracking & Model Registry (MLflow)

Every training run logs hyperparameter metrics (F1-score, Precision, Recall, ROC-AUC) to MLflow. The top-performing candidate model is automatically evaluated against the current registered production model before registry promotion.
3. Automated CI/CD Retraining Pipeline

A dedicated GitHub Actions workflow (retrain.yml) triggers on a weekly schedule or manual dispatch:

    Pulls tracked feature sets from DagsHub via DVC.

    Trains and evaluates candidate models.

    Registers the best model in MLflow.

    Automatically posts evaluation metrics as a commit comment.

4. Continuous Monitoring & Drift Detection

Integrated with Evidently AI (monitor.py) to periodically assess feature distribution changes between baseline training data and incoming production inference batches, generating HTML drift reports.
⚡ Getting Started (Local Development)
1. Clone the Repository
Bash

git clone https://github.com/ahmedhany-stack/cancelling_orders_detection.git
cd cancelling_orders_detection

2. Environment Setup
Bash

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

3. Pull Data via DVC
Bash

python -m dvc remote modify origin --local auth basic
python -m dvc remote modify origin --local user <DAGSHUB_USERNAME>
python -m dvc remote modify origin --local password <DAGSHUB_TOKEN>
python -m dvc pull -r origin

4. Run Model Training
Bash

python src/components/model_trainer.py

5. Launch Local FastAPI Web Server
Bash

python app.py

    Access Swagger UI Documentation at: http://localhost:8000/docs

🐳 Docker Deployment

To build and run the application using Docker:
Bash

# Build & Run with Docker Compose
docker-compose up --build -d

# Check application status
docker-compose ps

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.