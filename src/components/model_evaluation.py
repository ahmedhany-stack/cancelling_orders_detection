import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# استيراد mlflow
import mlflow

from src.config.configuration import PATHS
from src.exception.exception import CustomException
from src.logger.logger import logger
from src.utils.io import load_csv


class ModelEvaluation:
    """
    Model Evaluation Pipeline with MLflow Tracking
    """

    def __init__(self):
        self.data_path = Path(
            PATHS["artifacts"]["feature_data"]
        )

        self.model_path = Path(
            "artifacts/models/xgboost_model.pkl"
        )

        self.preprocessor_path = Path(
            "artifacts/preprocessors/preprocessor.pkl"
        )

        self.metrics_path = Path(
            "artifacts/models/evaluation_report.json"
        )

        self.threshold = 0.5
        
        # توجيه الـ MLflow إلى سيرفر DagsHub السحابي بدلاً من المجلد المحلي
        mlflow.set_tracking_uri("https://dagshub.com/ahmedhany-stack/cancelling_orders_detection.mlflow")
        
        # إعداد اسم التجربة لتطابق تجربة التدريب
        mlflow.set_experiment("Cancelling_Orders_Detection")

    # =====================================================
    # Load Artifacts
    # =====================================================
    def load_artifacts(self):
        logger.info("Loading Model And Preprocessor")
        model = joblib.load(self.model_path)
        preprocessor = joblib.load(self.preprocessor_path)
        return model, preprocessor

    # =====================================================
    # Prepare Test Data
    # =====================================================
    def prepare_test_data(self, df, preprocessor):
        logger.info("Preparing Test Data")

        X = df.drop(columns=["Cancelled"])
        y = df["Cancelled"]

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        X_test = preprocessor.transform(X_test)
        return X_test, y_test

    # =====================================================
    # Evaluation
    # =====================================================
    def evaluate_model(self, model, X_test, y_test):
        logger.info("Evaluating Model")

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        metrics = {
            "threshold": self.threshold,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions)),
            "recall": float(recall_score(y_test, predictions)),
            "f1_score": float(f1_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities))
        }

        # تسجيل المقاييس الأساسية في MLflow
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "threshold"})
        mlflow.log_param("evaluation_threshold", self.threshold)

        # إضافة المصفوفة والتقرير للقاموس لحفظهم محلياً
        metrics["confusion_matrix"] = cm.tolist()
        metrics["classification_report"] = report

        return metrics

    # =====================================================
    # Feature Importance
    # =====================================================
    def save_feature_importance(self, model, preprocessor):
        logger.info("Saving Feature Importance")

        try:
            feature_names = preprocessor.get_feature_names_out()

            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            })

            importance = importance.sort_values(by="Importance", ascending=False)
            path = Path("artifacts/models/feature_importance.csv")

            importance.to_csv(path, index=False)
            logger.info(f"Feature Importance Saved : {path}")

            # تسجيل ملف أهمية الميزات كـ Artifact في MLflow
            mlflow.log_artifact(str(path), artifact_path="evaluation")

        except Exception as e:
            logger.warning(f"Feature Importance Failed : {e}")

    # =====================================================
    # Main Pipeline
    # =====================================================
    def initiate_model_evaluation(self):
        try:
            logger.info("=" * 70)
            logger.info("Starting Model Evaluation with MLflow Tracking")

            # بدء Run جديدة مخصصة لعملية التقييم
            with mlflow.start_run(run_name="model_evaluation_run"):
                df = load_csv(self.data_path)
                model, preprocessor = self.load_artifacts()

                X_test, y_test = self.prepare_test_data(df, preprocessor)
                
                # تسجيل حجم بيانات الاختبار في الـ Tags
                mlflow.set_tag("test_set_shape", str(X_test.shape))

                metrics = self.evaluate_model(model, X_test, y_test)

                self.save_feature_importance(model, preprocessor)

                self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self.metrics_path, "w") as file:
                    json.dump(metrics, file, indent=4)

                # تسجيل ملف التقرير الكامل (يحتوي على الـ Confusion Matrix والتقرير التفصيلي)
                mlflow.log_artifact(str(self.metrics_path), artifact_path="evaluation")

                logger.info(f"Evaluation Report Saved : {self.metrics_path}")
                logger.info("Model Evaluation Completed Successfully")
                logger.info("=" * 70)

                return metrics

        except Exception as e:
            logger.exception("Model Evaluation Failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    evaluation = ModelEvaluation()
    evaluation.initiate_model_evaluation()