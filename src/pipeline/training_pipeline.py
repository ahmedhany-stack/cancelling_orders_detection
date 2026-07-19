import sys
from src.exception.exception import CustomException
from src.logger.logger import logger

# استيراد المكونات التي قمت بإنشائها
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher


class TrainingPipeline:
    """
    Training Pipeline Coordinator
    يقوم بإدارة وتشغيل مراحل التدريب بالكامل بالتوالي
    """
    
    def __init__(self):
        # تعريف كائنات المراحل المختلفة
        self.feature_engineering = FeatureEngineering()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()
        self.model_pusher = ModelPusher()

    def run_pipeline(self):
        try:
            logger.info("=" * 80)
            logger.info(">>> STARTING FULL TRAINING PIPELINE <<<")
            logger.info("=" * 80)

            # ------------------------------------------------------
            # 1. Feature Engineering Phase
            # ------------------------------------------------------
            logger.info("Executing Phase 1: Feature Engineering")
            # يقوم بقراءة البيانات الخام وتحويلها وحفظها في المسار المحدد (feature_data)
            self.feature_engineering.initiate_feature_engineering()
            logger.info("Phase 1: Feature Engineering Completed Successfully.\n")

            # ------------------------------------------------------
            # 2. Model Training Phase
            # ------------------------------------------------------
            logger.info("Executing Phase 2: Model Training")
            # يقوم بقراءة الـ feature_data، تصفية المتشابهات، تطبيق SMOTE، وتدريب النموذج وحفظ الـ Artifacts
            self.model_trainer.initiate_model_training()
            logger.info("Phase 2: Model Training Completed Successfully.\n")

            # ------------------------------------------------------
            # 3. Model Evaluation Phase
            # ------------------------------------------------------
            logger.info("Executing Phase 3: Model Evaluation")
            # يقوم بتقييم النموذج وحساب مقاييس الأداء وحفظ تقرير التقييم وأهمية الميزات
            self.model_evaluation.initiate_model_evaluation()
            logger.info("Phase 3: Model Evaluation Completed Successfully.\n")

            # ------------------------------------------------------
            # 4. Model Pusher Phase
            # ------------------------------------------------------
            logger.info("Executing Phase 4: Model Pusher")
            # ينقل النموذج والـ Preprocessor الجاهزين إلى مجلد الإنتاج النهائي
            pusher_artifacts = self.model_pusher.initiate_model_pusher()
            logger.info(f"Phase 4: Model Pusher Completed. Model deployed to: {pusher_artifacts['production_model_path']}\n")

            logger.info("=" * 80)
            logger.info(">>> FULL TRAINING PIPELINE COMPLETED SUCCESSFULLY <<<")
            logger.info("=" * 80)

        except Exception as e:
            logger.exception("Training Pipeline Failed in one of the phases!")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # لتشغيل الـ Pipeline بالكامل من سطر الأوامر
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()