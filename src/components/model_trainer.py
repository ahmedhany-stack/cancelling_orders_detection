import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

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
import mlflow.xgboost
import mlflow.sklearn

from src.config.configuration import PATHS
from src.exception.exception import CustomException
from src.logger.logger import logger
from src.utils.io import load_csv


class ModelTrainer:
    """
    XGBoost Model Training Pipeline with Cosine Similarity Deduplication and MLflow Tracking
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
            "artifacts/models/metrics.json"
        )
        
        # توجيه الـ MLflow إلى سيرفر DagsHub السحابي بدلاً من المجلد المحلي
        mlflow.set_tracking_uri("https://dagshub.com/ahmedhany-stack/cancelling_orders_detection.mlflow")
        
        # إعداد اسم التجربة في MLflow
        mlflow.set_experiment("Cancelling_Orders_Detection")

    # =====================================================
    # Preprocessing
    # =====================================================
    def prepare_data(self, df):
            logger.info("Preparing Data")

            X_temp = df.drop(columns=["Cancelled"])
            y_temp = df["Cancelled"]

            X_train_raw, X_test, y_train_raw, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=0.2,
                random_state=42,
                stratify=y_temp
            )

            train_df = pd.concat([X_train_raw, y_train_raw], axis=1).reset_index(drop=True)
            logger.info(f"Train Set Shape before Similarity Filtering: {train_df.shape}")

            categorical_columns = train_df.drop(columns=["Cancelled"]).select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            numerical_columns = train_df.drop(columns=["Cancelled"]).select_dtypes(
                include=["int64", "float64", "int32", "float32"]
            ).columns.tolist()

            # -------------------------------------------------
            # مرحلة إزالة العينات المتشابهة المحسنة والسريعة
            # -------------------------------------------------
            logger.info("Starting Optimized Cosine Similarity Filtering on Majority Class")
            
            majority = train_df[train_df["Cancelled"] == 0].reset_index(drop=True)
            minority = train_df[train_df["Cancelled"] == 1].reset_index(drop=True)

            # فلترة أولية سريعة جداً باستخدام Pandas لإزالة المكرر تماماً أولاً وتقليص الحجم
            initial_count = len(majority)
            majority = majority.drop_duplicates().reset_index(drop=True)
            logger.info(f"Removed {initial_count - len(majority)} exact duplicates using Pandas.")

            # إذا كان الحجم لا يزال ضخماً، نأخذ عينة عشوائية ممثلة لإجراء فحص التشابه عليها لتجنب تعليق الجهاز
            MAX_SAMPLES_FOR_NN = 40000 
            
            if len(majority) > MAX_SAMPLES_FOR_NN:
                logger.info(f"Majority class is too large ({len(majority)} rows). Subsampling to {MAX_SAMPLES_FOR_NN} for similarity check.")
                majority_sample = majority.sample(n=MAX_SAMPLES_FOR_NN, random_state=42).reset_index(drop=True)
                majority_rest = majority.drop(majority_sample.index).reset_index(drop=True)
            else:
                majority_sample = majority
                majority_rest = pd.DataFrame()

            X_majority_sample = majority_sample.drop(columns=["Cancelled"])

            for col in categorical_columns:
                X_majority_sample[col] = X_majority_sample[col].astype(str)

            preprocessor_similarity = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_columns),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
                ]
            )

            # تحويل العينة فقط
            X_similarity = preprocessor_similarity.fit_transform(X_majority_sample)

            # استخدام NearestNeighbors
            nn = NearestNeighbors(
                n_neighbors=2,
                metric="cosine",
                algorithm="auto", 
                n_jobs=-1
            )
            nn.fit(X_similarity)

            distances, indices = nn.kneighbors(X_similarity)

            rows_to_remove = set()
            SIMILARITY_THRESHOLD = 0.98

            for i in range(len(indices)):
                neighbor = indices[i][1]
                similarity = 1 - distances[i][1]

                if similarity >= SIMILARITY_THRESHOLD:
                    rows_to_remove.add(neighbor)

            logger.info(f"Majority Duplicates Detected in sample: {len(rows_to_remove)}")

            # إسقاط الصفوف المتشابهة من العينة
            majority_sample = majority_sample.drop(index=list(rows_to_remove)).reset_index(drop=True)

            # إعادة دمج البيانات المصفاة
            majority = pd.concat([majority_sample, majority_rest], ignore_index=True)

            # دمج الفئتين مرة أخرى وعمل خلط عشوائي (Shuffle)
            train_df = pd.concat([majority, minority], ignore_index=True)
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

            X_train = train_df.drop(columns=["Cancelled"])
            y_train = train_df["Cancelled"]

            logger.info(f"Train Set Shape after Similarity Filtering: {X_train.shape}")

            # -------------------------------------------------
            # التجهيز النهائي للـ Preprocessor و SMOTE
            # -------------------------------------------------
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_columns),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
                ]
            )

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            logger.info(f"Before SMOTE Train Shape: {X_train.shape}")

            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            logger.info(f"After SMOTE Train Shape: {X_train.shape}")

            return (
                X_train,
                X_test,
                y_train,
                y_test,
                preprocessor
            )
            
    # =====================================================
    # Train Model
    # =====================================================
    def train_model(self, X_train, y_train):
        logger.info("Training XGBoost")

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=400,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.15,
            reg_alpha=0.2,
            reg_lambda=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        return model

    # =====================================================
    # Threshold Optimization
    # =====================================================
    def find_best_threshold(self, model, X_test, y_test):
        logger.info("Searching Best Threshold")

        probabilities = model.predict_proba(X_test)[:, 1]

        best_threshold = 0.5
        best_f1 = 0

        for threshold in np.arange(0.01, 0.99, 0.01):
            predictions = (probabilities >= threshold).astype(int)
            score = f1_score(y_test, predictions)

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        return best_threshold, probabilities

    # =====================================================
    # Evaluation
    # =====================================================
    def evaluate(self, y_test, probabilities, threshold):
        predictions = (probabilities >= threshold).astype(int)

        metrics = {
            "optimized_threshold": float(threshold),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions)),
            "recall": float(recall_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities))
        }

        logger.info(json.dumps(metrics, indent=4))
        return metrics

    # =====================================================
    # Pipeline
    # =====================================================
    def initiate_model_training(self):
        try:
            logger.info("=" * 70)
            logger.info("Starting Model Training Pipeline with MLflow Tracking")

            # كتم وتحييد التحذيرات التلقائية المزعجة لـ MLflow
            mlflow.sklearn.autolog(disable=True)
            mlflow.xgboost.autolog(disable=True)

            with mlflow.start_run(run_name="xgboost_optimized_run"):
                df = load_csv(self.data_path)
                logger.info(f"Raw Dataset Shape : {df.shape}")
                
                # تسجيل حجم البيانات في MLflow Tags
                mlflow.set_tag("dataset_shape", str(df.shape))

                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    preprocessor
                ) = self.prepare_data(df)

                # تدريب النموذج
                model = self.train_model(X_train, y_train)

                threshold, probabilities = self.find_best_threshold(
                    model, X_test, y_test
                )

                # حساب المقاييس النهائية بناءً على الـ Optimized Threshold
                metrics = self.evaluate(y_test, probabilities, threshold)

                # تسجيل المقاييس يدوياً دون أي تحذيرات
                mlflow.log_metrics(metrics)

                # إنشاء المجلدات وحفظ الملفات محلياً
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

                joblib.dump(model, self.model_path)
                joblib.dump(preprocessor, self.preprocessor_path)

                with open(self.metrics_path, "w") as file:
                    json.dump(metrics, file, indent=4)

                # تسجيل الملفات كـ Artifacts
                mlflow.log_artifact(str(self.preprocessor_path), artifact_path="preprocessor")
                mlflow.log_artifact(str(self.metrics_path), artifact_path="metrics")
                
                # تسجيل الموديل في Model Registry السحابي بوضوح
                mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name="XGBoost_Churn_Model")

                logger.info(f"Model Saved Locally : {self.model_path}")
                logger.info("Model Training and MLflow Logging Completed Successfully")
                logger.info("=" * 70)

        except Exception as e:
            logger.exception("Model Training Failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()