import sys
from pathlib import Path

from src.config.configuration import PATHS
from src.exception.exception import CustomException
from src.logger.logger import logger
from src.utils.file_system import create_directories
from src.utils.io import load_excel, save_csv


class DataIngestion:
    """
    Responsible for loading the raw dataset
    and saving it inside the artifacts directory.
    """

    def __init__(self):

        self.data_path = Path(PATHS["data"]["raw"])

        self.artifacts_dir = Path(PATHS["artifacts"]["root"])

        self.raw_data_path = Path(PATHS["artifacts"]["raw_data"])

    def initiate_data_ingestion(self) -> Path:

        logger.info("=" * 60)
        logger.info("Starting Data Ingestion")

        try:

            logger.info("Creating Artifacts Directory")
            create_directories(self.artifacts_dir)

            if not self.data_path.exists():

                raise FileNotFoundError(
                    f"Dataset not found: {self.data_path}"
                )

            logger.info(f"Loading Dataset: {self.data_path}")

            df = load_excel(self.data_path)

            logger.info(f"Dataset Loaded Successfully | Shape: {df.shape}")

            logger.info(f"Saving Raw Dataset -> {self.raw_data_path}")

            save_csv(df, self.raw_data_path)

            logger.info("Data Ingestion Completed Successfully")
            logger.info("=" * 60)

            return self.raw_data_path

        except Exception as e:

            logger.exception("Data Ingestion Failed")

            raise CustomException(e, sys)


if __name__ == "__main__":

    ingestion = DataIngestion()

    raw_path = ingestion.initiate_data_ingestion()

    print(f"Raw Data Saved At: {raw_path}")