import sys
import dagshub
import mlflow
from src.exception.exception import CustomException
from src.logger.logger import logger

# استيراد المكونات
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from monitor import run_monitoring  # استيراد سكريبت المراقبة

# إعداد DagsHub و MLflow
dagshub.init(repo_owner='ahmedhany-stack', repo_name='cancelling_orders_detection', mlflow=True)
mlflow.autolog()

class TrainingPipeline:
    """
    Training Pipeline Coordinator
    تدير مراحل التدريب بالكامل مع المراقبة التلقائية
    """
    
    def __init__(self):
        self.feature_engineering = FeatureEngineering()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()
        self.model_pusher = ModelPusher()

    def run_pipeline(self):
        try:
            logger.info("=" * 80)
            logger.info(">>> STARTING FULL TRAINING PIPELINE <<<")
            logger.info("=" * 80)

            # 1. Feature Engineering
            logger.info("Executing Phase 1: Feature Engineering")
            self.feature_engineering.initiate_feature_engineering()
            
            # 2. Model Training
            logger.info("Executing Phase 2: Model Training")
            self.model_trainer.initiate_model_training()
            
            # 3. Model Evaluation
            logger.info("Executing Phase 3: Model Evaluation")
            self.model_evaluation.initiate_model_evaluation()
            
            # 4. Model Pusher
            logger.info("Executing Phase 4: Model Pusher")
            pusher_artifacts = self.model_pusher.initiate_model_pusher()
            
            # 5. Automated Monitoring (إضافة احترافية)
            logger.info("Executing Phase 5: Automated Data Monitoring")
            run_monitoring() # هذا السكريبت سيقوم بالتحليل ويرسل تنبيه Slack إذا وجد Drift
            
            # ربط التقرير بـ MLflow ليظهر في الـ Dashboard
            with mlflow.start_run(nested=True):
                mlflow.log_artifact("reports/drift_report.html")
                logger.info("Monitoring report logged to MLflow successfully.")

            logger.info("=" * 80)
            logger.info(">>> FULL TRAINING PIPELINE COMPLETED SUCCESSFULLY <<<")
            logger.info("=" * 80)

        except Exception as e:
            logger.exception("Training Pipeline Failed!")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()