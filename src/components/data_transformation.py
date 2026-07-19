import sys
from pathlib import Path

import pandas as pd
import numpy as np


from src.config.configuration import PATHS
from src.exception.exception import CustomException
from src.logger.logger import logger
from src.utils.io import load_csv



class DataTransformation:
    """
    Data Transformation Component

    Responsible for:
    - Handling missing values
    - Fixing data types
    - Removing invalid records
    - Saving clean dataset
    """



    def __init__(self):

        self.input_path = Path(
            PATHS["artifacts"]["raw_data"]
        )


        self.output_path = Path(
            PATHS["artifacts"]["clean_data"]
        )



    def handle_missing_values(self, df):

        logger.info(
            "Handling Missing Values"
        )


        # Description
        if "Description" in df.columns:

            df["Description"] = (
                df["Description"]
                .fillna("Unknown")
            )


        # Customer ID
        if "Customer ID" in df.columns:

            df["Customer ID"] = (
                df["Customer ID"]
                .fillna(-1)
            )


        return df



    def fix_data_types(self, df):

        logger.info(
            "Fixing Data Types"
        )


        if "InvoiceDate" in df.columns:

            df["InvoiceDate"] = pd.to_datetime(
                df["InvoiceDate"],
                errors="coerce"
            )


        numeric_columns = [

            "Quantity",
            "Price",
            "Customer ID"

        ]


        for col in numeric_columns:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )


        return df



    def remove_invalid_records(self, df):

        logger.info(
            "Removing Invalid Records"
        )


        # Remove rows without invoice
        if "Invoice" in df.columns:

            df = df.dropna(
                subset=["Invoice"]
            )


        # Remove invalid dates
        if "InvoiceDate" in df.columns:

            df = df[
                df["InvoiceDate"].notna()
            ]


        # Remove zero price
        if "Price" in df.columns:

            df = df[
                df["Price"] > 0
            ]


        # Remove zero quantity
        if "Quantity" in df.columns:

            df = df[
                df["Quantity"] != 0
            ]


        return df



    def remove_duplicates(self, df):

        logger.info(
            "Removing Duplicate Rows"
        )


        before = df.shape[0]


        df = df.drop_duplicates()


        after = df.shape[0]


        logger.info(
            f"Removed Duplicates : {before-after}"
        )


        return df



    def create_basic_features(self, df):

        """
        Only basic features.
        Advanced features are created
        in Feature Engineering.
        """

        logger.info(
            "Creating Basic Features"
        )


        df["Invoice_Total"] = (

            df["Quantity"] *
            df["Price"]

        )


        df["Cancelled"] = (

            df["Invoice"]
            .astype(str)
            .str.startswith("C")
            .astype(int)

        )


        return df



    def initiate_data_transformation(self):

        try:

            logger.info("="*70)

            logger.info(
                "Starting Data Transformation"
            )


            df = load_csv(
                self.input_path
            )


            logger.info(
                f"Raw Shape : {df.shape}"
            )



            df = self.handle_missing_values(
                df
            )


            df = self.fix_data_types(
                df
            )


            df = self.remove_invalid_records(
                df
            )


            df = self.remove_duplicates(
                df
            )


            df = self.create_basic_features(
                df
            )



            self.output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            df.to_csv(
                self.output_path,
                index=False
            )



            logger.info(
                f"Clean Data Saved : {self.output_path}"
            )


            logger.info(
                f"Final Shape : {df.shape}"
            )


            logger.info(
                "Data Transformation Completed Successfully"
            )


            logger.info("="*70)



            return df



        except Exception as e:


            logger.exception(
                "Data Transformation Failed"
            )


            raise CustomException(
                e,
                sys
            )




if __name__ == "__main__":


    transformation = DataTransformation()


    transformation.initiate_data_transformation()