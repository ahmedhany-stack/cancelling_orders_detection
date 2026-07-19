import sys
import shutil
from pathlib import Path

from src.config.configuration import PATHS
from src.exception.exception import CustomException
from src.logger.logger import logger


class ModelPusher:
    """
    Model Pusher Pipeline
    يقوم بنقل النموذج المعتمد والـ Preprocessor إلى بيئة الإنتاج (Saved Models)
    """

    def __init__(self):
        # مسار النموذج والـ Preprocessor الذي تم تدريبهما في المجلد المؤقت (Artifacts)
        self.trained_model_path = Path("artifacts/models/xgboost_model.pkl")
        self.trained_preprocessor_path = Path("artifacts/preprocessors/preprocessor.pkl")

        # مسارات بيئة الإنتاج (حيث يقرأ تطبيق الـ API أو الويب النسخة المستقرة مباشرة)
        self.production_model_dir = Path("saved_models")
        self.production_preprocessor_dir = Path("saved_preprocessors")

        # تسمية الملفات النهائية في الإنتاج
        self.production_model_path = self.production_model_dir / "model.pkl"
        self.production_preprocessor_path = self.production_preprocessor_dir / "preprocessor.pkl"


    def initiate_model_pusher(self):
        try:
            logger.info("=" * 70)
            logger.info("Starting Model Pusher Pipeline")

            # 1. التأكد من وجود المجلدات المخصصة للإنتاج، وإنشائها إن لم تكن موجودة
            self.production_model_dir.mkdir(parents=True, exist_ok=True)
            self.production_preprocessor_dir.mkdir(parents=True, exist_ok=True)

            # 2. التحقق من وجود الملفات المدربة حديثاً في الـ Artifacts قبل محاولة نقلها
            if not self.trained_model_path.exists():
                raise FileNotFoundError(f"Trained model not found at {self.trained_model_path}")
            
            if not self.trained_preprocessor_path.exists():
                raise FileNotFoundError(f"Trained preprocessor not found at {self.trained_preprocessor_path}")

            # 3. دفع النموذج (Model) إلى الإنتاج
            logger.info(f"Pushing trained model from {self.trained_model_path} to {self.production_model_path}")
            shutil.copy(src=self.trained_model_path, dst=self.production_model_path)

            # 4. دفع معالج البيانات (Preprocessor) إلى الإنتاج
            logger.info(f"Pushing preprocessor from {self.trained_preprocessor_path} to {self.production_preprocessor_path}")
            shutil.copy(src=self.trained_preprocessor_path, dst=self.production_preprocessor_path)

            logger.info("Model and Preprocessor successfully pushed to Production environment!")
            logger.info("Model Pusher Completed Successfully")
            logger.info("=" * 70)

            return {
                "production_model_path": str(self.production_model_path),
                "production_preprocessor_path": str(self.production_preprocessor_path)
            }

        except Exception as e:
            logger.exception("Model Pusher Failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pusher = ModelPusher()
    pusher.initiate_model_pusher()