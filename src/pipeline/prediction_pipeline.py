import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from src.exception.exception import CustomException
from src.logger.logger import logger
from src.components.feature_engineering import FeatureEngineering


class PredictionPipeline:
    """
    Prediction Pipeline
    مسؤول عن استقبال الأعمدة الـ 8 الأساسية، وتمريرها عبر كلاس هندسة الميزات بالكامل،
    ثم عمل التوقعات باستخدام النموذج المدرب.
    """

    def __init__(self):
        self.model_path = Path("saved_models/model.pkl")
        self.preprocessor_path = Path("saved_preprocessors/preprocessor.pkl")

    def predict(self, raw_features: pd.DataFrame):
        """
        تستقبل الأعمدة الـ 8 الخام، تقوم بتحويلها وبناء كافة الميزات بالترتيب الصحيح، ثم تتوقع النتيجة.
        """
        try:
            logger.info("Starting prediction process on raw customer data")

            # 1. التحقق من وجود ملفات النموذج والمعالج
            if not self.model_path.exists() or not self.preprocessor_path.exists():
                raise FileNotFoundError(
                    "Production model or preprocessor not found! Run the Training Pipeline first."
                )

            # تأكيد أنواع البيانات الأساسية قبل البدء
            df = raw_features.copy()
            df["Quantity"] = df["Quantity"].astype(int)
            df["Price"] = df["Price"].astype(float)
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

            # 2. استدعاء خطوات الـ Feature Engineering بالترتيب الصحيح والكامل
            logger.info("Applying full Feature Engineering pipeline on prediction data")
            fe = FeatureEngineering()
            
            df = fe.create_date_features(df)
            df = fe.create_transaction_features(df)
            df = fe.create_customer_features(df)
            df = fe.create_product_features(df)
            df = fe.create_behavior_features(df)
            df = fe.create_advanced_features(df)
            
            # 3. حذف الأعمدة غير المستخدمة (مثل Invoice ID, Quantity المباشرة... إلخ) التي لا يتوقعها الـ preprocessor
            df = fe.remove_unused_columns(df)

            # 4. تحميل النموذج والمعالج
            logger.info("Loading production model and preprocessor")
            model = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)

            # 5. معالجة البيانات بواسطة الـ Preprocessor
            logger.info("Preprocessing engineered features")
            scaled_features = preprocessor.transform(df)

            # 6. عمل التوقعات
            logger.info("Generating final predictions")
            predictions = model.predict(scaled_features)
            probabilities = model.predict_proba(scaled_features)[:, 1]

            results = pd.DataFrame({
                "Prediction": predictions,
                "Cancel_Probability": probabilities
            })

            logger.info("Prediction pipeline finished successfully")
            return results

        except Exception as e:
            logger.exception("Error occurred during prediction pipeline execution")
            raise CustomException(e, sys)


class CustomData:
    """
    كلاس مساعد يستقبل فقط الأعمدة الـ 8 المتوفرة لديك من المستخدم
    ويحولها إلى DataFrame متناسق مع كود المعالجة.
    """

    def __init__(
        self,
        invoice: str,
        stock_code: str,
        description: str,
        quantity: int,
        invoice_date: str,
        price: float,
        customer_id: str,
        country: str
    ):
        self.invoice = invoice
        self.stock_code = stock_code
        self.description = description
        self.quantity = quantity
        self.invoice_date = invoice_date
        self.price = price
        self.customer_id = customer_id
        self.country = country

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "Invoice": [self.invoice],
                "StockCode": [self.stock_code],
                "Description": [self.description],
                "Quantity": [self.quantity],
                "InvoiceDate": [pd.to_datetime(self.invoice_date)],
                "Price": [self.price],
                "Customer ID": [self.customer_id],
                "Country": [self.country]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logger.info("Raw input data successfully structured into DataFrame")
            return df

        except Exception as e:
            logger.exception("Failed to structure raw input data")
            raise CustomException(e, sys)