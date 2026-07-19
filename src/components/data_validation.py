import sys
import json
import pandas as pd
from pathlib import Path

from src.config.configuration import PATHS
from src.exception.exception import CustomException
from src.logger.logger import logger
from src.utils.io import load_csv


class DataValidation:
    """
    Validate raw dataset before data transformation.
    """

    def __init__(self):

        self.raw_data_path = Path(
            PATHS["artifacts"]["raw_data"]
        )

        self.validation_report_path = Path(
            PATHS["artifacts"].get(
                "validation_report",
                "artifacts/validation_report.json"
            )
        )


        self.expected_columns = [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country"
        ]


        self.expected_dtypes = {

            "Quantity": "numeric",
            "Price": "numeric"

        }


        self.min_rows = 1000



    def initiate_data_validation(self) -> bool:


        logger.info("=" * 70)
        logger.info("Starting Data Validation")


        report = {}


        try:


            # -------------------------------------------------
            # Check File Exists
            # -------------------------------------------------

            if not self.raw_data_path.exists():

                raise FileNotFoundError(
                    f"Dataset not found : {self.raw_data_path}"
                )


            logger.info(
                "Dataset Found"
            )



            # -------------------------------------------------
            # Load Dataset
            # -------------------------------------------------

            df = load_csv(
                self.raw_data_path
            )


            logger.info(
                f"Dataset Loaded Successfully | Shape : {df.shape}"
            )


            report["rows"] = int(df.shape[0])
            report["columns"] = int(df.shape[1])



            # -------------------------------------------------
            # Empty Dataset
            # -------------------------------------------------

            if df.empty:

                raise ValueError(
                    "Dataset is Empty"
                )


            logger.info(
                "Dataset is Not Empty"
            )



            # -------------------------------------------------
            # Minimum Dataset Size
            # -------------------------------------------------

            if len(df) < self.min_rows:

                raise ValueError(
                    f"Dataset size too small : {len(df)}"
                )



            # -------------------------------------------------
            # Validate Columns
            # -------------------------------------------------

            missing_columns = [

                col
                for col in self.expected_columns
                if col not in df.columns

            ]


            if missing_columns:

                raise ValueError(
                    f"Missing Columns : {missing_columns}"
                )


            logger.info(
                "All Required Columns Exist"
            )



            # -------------------------------------------------
            # Duplicate Columns
            # -------------------------------------------------

            duplicated_columns = (
                df.columns[df.columns.duplicated()]
            )


            if len(duplicated_columns) > 0:

                raise ValueError(
                    f"Duplicate Columns : {list(duplicated_columns)}"
                )


            logger.info(
                "No Duplicate Columns"
            )



            # -------------------------------------------------
            # Missing Values
            # -------------------------------------------------

            missing_values = (
                df.isnull()
                .sum()
                .to_dict()
            )


            report["missing_values"] = {

                k: int(v)
                for k, v in missing_values.items()

            }


            total_missing = sum(
                missing_values.values()
            )


            if total_missing > 0:


                logger.warning(
                    "Dataset Contains Missing Values"
                )


                for col, count in missing_values.items():

                    if count > 0:

                        logger.warning(
                            f"{col} : {count}"
                        )


            else:

                logger.info(
                    "No Missing Values"
                )



            # -------------------------------------------------
            # Data Types Validation
            # -------------------------------------------------

            for column, dtype in self.expected_dtypes.items():


                if dtype == "numeric":


                    if not pd.api.types.is_numeric_dtype(
                        df[column]
                    ):

                        raise ValueError(
                            f"{column} should be numeric"
                        )


            logger.info(
                "Data Types Validation Passed"
            )



            # -------------------------------------------------
            # Duplicate Rows
            # -------------------------------------------------

            duplicate_rows = (
                df.duplicated()
                .sum()
            )


            report["duplicate_rows"] = int(
                duplicate_rows
            )


            logger.info(
                f"Duplicate Rows : {duplicate_rows}"
            )



            # -------------------------------------------------
            # Business Rules
            # -------------------------------------------------

            # Quantity
            # Negative values mean cancellation
            # in Online Retail Dataset

            negative_quantity = df[
                df["Quantity"] <= 0
            ]


            report["negative_quantity_count"] = int(
                len(negative_quantity)
            )


            logger.warning(
                f"Negative Quantity Rows : {len(negative_quantity)}"
            )



            # Price Validation

            invalid_price = df[
                df["Price"] <= 0
            ]


            report["invalid_price_count"] = int(
                len(invalid_price)
            )


            if len(invalid_price) > 0:

                logger.warning(
                    f"Invalid Price Rows : {len(invalid_price)}"
                )

            else:

                logger.info(
                    "No Invalid Price Values"
                )


            logger.info(
                "Business Rules Validation Passed"
            )
                        # -------------------------------------------------
            # Invoice Format Validation
            # -------------------------------------------------

            invalid_invoice = df[
                ~df["Invoice"]
                .astype(str)
                .str.match(r"^[A-Z0-9]+$")
            ]


            report["invalid_invoice_count"] = int(
                len(invalid_invoice)
            )


            if len(invalid_invoice) > 0:

                raise ValueError(
                    "Invalid Invoice Format Found"
                )


            logger.info(
                "Invoice Format Validation Passed"
            )



            # -------------------------------------------------
            # Unique Values Checks
            # -------------------------------------------------

            report["unique_values"] = {

                "country":
                int(df["Country"].nunique()),


                "description":
                int(df["Description"].nunique()),


                "customers":
                int(df["Customer ID"].nunique())

            }


            logger.info(
                f"Countries : {df['Country'].nunique()}"
            )


            logger.info(
                f"Descriptions : {df['Description'].nunique()}"
            )


            logger.info(
                f"Customers : {df['Customer ID'].nunique()}"
            )



            # -------------------------------------------------
            # Outlier Report
            # -------------------------------------------------

            outlier_report = {}


            for col in [
                "Quantity",
                "Price"
            ]:


                Q1 = df[col].quantile(0.25)

                Q3 = df[col].quantile(0.75)

                IQR = Q3 - Q1


                outliers = df[

                    (df[col] < Q1 - 1.5 * IQR)
                    |
                    (df[col] > Q3 + 1.5 * IQR)

                ]


                outlier_report[col] = int(
                    len(outliers)
                )


            report["outliers"] = outlier_report


            logger.info(
                f"Outliers : {outlier_report}"
            )



            # -------------------------------------------------
            # Final Status
            # -------------------------------------------------

            report["status"] = "PASSED"



            # -------------------------------------------------
            # Save Validation Report
            # -------------------------------------------------

            self.validation_report_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            with open(
                self.validation_report_path,
                "w"
            ) as file:


                json.dump(
                    report,
                    file,
                    indent=4
                )



            logger.info(
                f"Validation Report Saved : {self.validation_report_path}"
            )


            logger.info(
                "Data Validation Completed Successfully"
            )


            logger.info("=" * 70)


            return True



        except Exception as e:


            report["status"] = "FAILED"


            self.validation_report_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            with open(
                self.validation_report_path,
                "w"
            ) as file:


                json.dump(
                    report,
                    file,
                    indent=4
                )



            logger.exception(
                "Data Validation Failed"
            )


            raise CustomException(
                e,
                sys
            )



if __name__ == "__main__":


    validation = DataValidation()

    validation.initiate_data_validation()