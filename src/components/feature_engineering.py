import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


from src.config.configuration import PATHS
from src.exception.exception import CustomException
from src.logger.logger import logger
from src.utils.io import load_csv



class FeatureEngineering:
    """
    Feature Engineering Pipeline
    for Online Retail II Dataset.
    """



    def __init__(self):

        self.input_path = Path(
            PATHS["artifacts"]["clean_data"]
        )


        self.output_path = Path(
            PATHS["artifacts"].get(
                "feature_data",
                "artifacts/engineered_data.csv"
            )
        )



    # =====================================================
    # Date Features
    # =====================================================

    def create_date_features(self, df):

        logger.info(
            "Creating date features"
        )


        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )


        df["Year"] = (
            df["InvoiceDate"]
            .dt.year
        )


        df["Month"] = (
            df["InvoiceDate"]
            .dt.month
        )


        df["Day_Name"] = (
            df["InvoiceDate"]
            .dt.day_name()
        )


        df["Hour"] = (
            df["InvoiceDate"]
            .dt.hour
        )


        df["Week"] = (
            df["InvoiceDate"]
            .dt.isocalendar()
            .week
            .astype(int)
        )


        df["Quarter"] = (
            df["InvoiceDate"]
            .dt.quarter
        )




        return df



    # =====================================================
    # Transaction Features
    # =====================================================

    def create_transaction_features(self, df):

        logger.info(
            "Creating transaction features"
        )


        df["Cancelled"] = (

            df["Invoice"]
            .astype(str)
            .str.startswith("C")
            .astype(int)

        )







        df["Items_Count"] = (

            df.groupby("Invoice")
            ["StockCode"]
            .transform("count")

        )



        df["Unique_Products"] = (

            df.groupby("Invoice")
            ["StockCode"]
            .transform("nunique")

        )



        df["Average_Price"] = (

            df.groupby("Invoice")
            ["Price"]
            .transform("mean")

        )


        return df



    # =====================================================
    # Customer Features
    # =====================================================

    def create_customer_features(self, df):

        logger.info(
            "Creating customer features"
        )


        df = df.sort_values(
            [
                "Customer ID",
                "InvoiceDate"
            ]
        )


        df["Customer_Orders"] = (

            df.groupby("Customer ID")
            ["Invoice"]
            .transform("nunique")

        )


        df["Customer_Frequency"] = (

            df["Customer_Orders"]

        )






        df["Days_Since_Last_Order"] = (

            df.groupby("Customer ID")
            ["InvoiceDate"]
            .diff()
            .dt.days
            .fillna(0)

        )


        return df



    # =====================================================
    # Product Features
    # =====================================================

    def create_product_features(self, df):

        logger.info(
            "Creating product features"
        )


        df["Product_Avg_Price"] = (

            df.groupby("StockCode")
            ["Price"]
            .transform("mean")

        )


        df["Product_Frequency"] = (

            df.groupby("StockCode")
            ["StockCode"]
            .transform("count")

        )


        return df
    
        # =====================================================
    # Behavior Features
    # =====================================================

    def create_behavior_features(self, df):

        logger.info(
            "Creating behavior features"
        )


        def hour_category(hour):

            if 5 <= hour < 12:
                return "Morning"

            elif 12 <= hour < 17:
                return "Afternoon"

            return "Night"



        df["Hour_Category"] = (
            df["Hour"]
            .apply(hour_category)
        )



        df["Weekend"] = (

            df["Day_Name"]
            .isin(
                [
                    "Saturday",
                    "Sunday"
                ]
            )
            .astype(int)

        )



        def season(month):

            if month in [12, 1, 2]:

                return "Winter"

            elif month in [3, 4, 5]:

                return "Spring"

            elif month in [6, 7, 8]:

                return "Summer"

            return "Autumn"



        df["Season"] = (

            df["Month"]
            .apply(season)

        )



        df["Is_UK"] = (

            df["Country"]
            ==
            "United Kingdom"

        ).astype(int)



        return df



    # =====================================================
    # Advanced Features
        # =====================================================
    def create_advanced_features(self, df):
            logger.info("Creating advanced features")

            # ==========================================
            # 1. Calculate Customer_Value FIRST (To avoid KeyError)
            # ==========================================
            df["Customer_Value"] = (
                df["Customer_Orders"]
                *
                df["Average_Price"]
            )

            # ==========================================
            # 2. Additional Numerical Features
            # ==========================================

            df["Total_Items_Bought"] = (
                df.groupby("Customer ID")["Quantity"]
                .transform("sum")
            )

            df["Customer_Lifetime"] = (
                df.groupby("Customer ID")["InvoiceDate"]
                .transform(
                    lambda x: (x.max() - x.min()).days
                )
            )

            df["Items_Per_Order"] = (
                df["Total_Items_Bought"]
                /
                df["Customer_Orders"].replace(0, 1)
            )

            df["Products_Per_Order"] = (
                df["Unique_Products"]
                /
                df["Customer_Orders"].replace(0, 1)
            )

            df["Product_Diversity"] = (
                df["Unique_Products"]
                /
                df["Items_Count"].replace(0, 1)
            )

            df["Avg_Items_Per_Product"] = (
                df["Items_Count"]
                /
                df["Unique_Products"].replace(0, 1)
            )

            df["Spend_Per_Order"] = (
                df["Customer_Value"]
                /
                df["Customer_Orders"].replace(0, 1)
            )

            df["Spend_Per_Product"] = (
                df["Customer_Value"]
                /
                df["Unique_Products"].replace(0, 1)
            )

            # ==========================================
            # Price Category
            # ==========================================

            df["Price_Category"] = pd.cut(
                df["Price"],
                bins=[
                    -1,
                    2,
                    5,
                    10,
                    np.inf
                ],
                labels=[
                    "Low",
                    "Medium",
                    "High",
                    "Premium"
                ]
            )

            # ==========================
            # Safe Log Features
            # ==========================

            log_columns = [
                "Price",
                "Customer_Orders",
                "Items_Count",
                "Total_Items_Bought",
                "Unique_Products",
                "Average_Price",
                "Customer_Frequency",
                "Product_Frequency",
                "Days_Since_Last_Order",
                "Customer_Value",
                "Customer_Lifetime"
            ]

            for col in log_columns:
                if col in df.columns:
                    df[f"{col}_Log"] = np.log1p(
                        np.clip(
                            df[col],
                            a_min=0,
                            a_max=None
                        )
                    )

            df["High_Frequency_Customer"] = (
                df["Customer_Frequency"]
                >
                df["Customer_Frequency"].median()
            ).astype(int)

            df["Expensive_Product"] = (
                df["Price"]
                >
                df["Price"].median()
            ).astype(int)

            df["Loyal_Customer"] = (
                df["Customer_Orders"]
                >
                df["Customer_Orders"].quantile(0.75)
            ).astype(int)

            df["New_Customer"] = (
                df["Customer_Orders"]
                ==
                1
            ).astype(int)

            df["Rare_Product"] = (
                df["Product_Frequency"]
                <
                df["Product_Frequency"].quantile(0.25)
            ).astype(int)

            df["Popular_Product"] = (
                df["Product_Frequency"]
                >
                df["Product_Frequency"].quantile(0.75)
            ).astype(int)

            # ==========================================
            # Interaction Features
            # ==========================================

            df["Price_x_CustomerOrders"] = (
                df["Price"]
                *
                df["Customer_Orders"]
            )

            df["Price_x_ProductFreq"] = (
                df["Price"]
                *
                df["Product_Frequency"]
            )

            df["Orders_x_ProductFreq"] = (
                df["Customer_Orders"]
                *
                df["Product_Frequency"]
            )

            df["Orders_x_Items"] = (
                df["Customer_Orders"]
                *
                df["Items_Count"]
            )

            df["Items_x_Price"] = (
                df["Items_Count"]
                *
                df["Price"]
            )

            df["Items_x_ProductFreq"] = (
                df["Items_Count"]
                *
                df["Product_Frequency"]
            )

            df["AveragePrice_x_ProductFreq"] = (
                df["Average_Price"]
                *
                df["Product_Frequency"]
            )

            df["AveragePrice_x_Orders"] = (
                df["Average_Price"]
                *
                df["Customer_Orders"]
            )

            df["Order_to_Product_Ratio"] = (
                df["Customer_Orders"]
                /
                df["Product_Frequency"].replace(0, 1)
            )

            df["Price_to_ProductFreq"] = (
                df["Price"]
                /
                df["Product_Frequency"].replace(0, 1)
            )

            df["Frequency_to_Order"] = (
                df["Product_Frequency"]
                /
                df["Customer_Orders"].replace(0, 1)
            )

            df["Price_to_Items"] = (
                df["Price"]
                /
                df["Items_Count"].replace(0, 1)
            )

            return df

        # =====================================================
        # Remove Unused Columns
    # =====================================================

    def remove_unused_columns(self, df):

        logger.info(
            "Removing unused columns"
        )


        columns_to_drop = [

            "Invoice",

            "InvoiceDate",

            "Description",

            "Customer ID",
            "Invoice_Total",

            "Month",
            "Quantity"

        ]


        df.drop(

            columns=[
                col
                for col in columns_to_drop
                if col in df.columns
            ],

            inplace=True

        )


        return df



    # =====================================================
    # Save Feature Report
    # =====================================================

    def save_feature_report(self, df):

        report = {

            "rows": df.shape[0],

            "columns": df.shape[1],

            "features": list(df.columns)

        }



        report_path = (

            self.output_path.parent /
            "feature_report.json"

        )



        with open(
            report_path,
            "w"
        ) as file:


            json.dump(

                report,

                file,

                indent=4

            )



        logger.info(

            f"Feature Report Saved : {report_path}"

        )



    # =====================================================
    # Main Pipeline
    # =====================================================

    def initiate_feature_engineering(self):

        try:


            logger.info("=" * 70)

            logger.info(
                "Starting Feature Engineering"
            )



            if not self.input_path.exists():


                raise FileNotFoundError(

                    f"Clean Data Not Found : {self.input_path}"

                )



            df = load_csv(

                self.input_path

            )



            logger.info(

                f"Input Shape : {df.shape}"

            )



            df = self.create_date_features(df)


            df = self.create_transaction_features(df)


            df = self.create_customer_features(df)


            df = self.create_product_features(df)


            df = self.create_behavior_features(df)


            df = self.create_advanced_features(df)


            df = self.remove_unused_columns(df)



            self.output_path.parent.mkdir(

                parents=True,

                exist_ok=True

            )



            df.to_csv(

                self.output_path,

                index=False

            )



            self.save_feature_report(df)



            logger.info(

                f"Feature Data Saved : {self.output_path}"

            )


            logger.info(

                f"Final Shape : {df.shape}"

            )


            logger.info(

                "Feature Engineering Completed Successfully"

            )


            logger.info("=" * 70)



            return df



        except Exception as e:


            logger.exception(

                "Feature Engineering Failed"

            )


            raise CustomException(

                e,

                sys

            )





# =====================================================
# Run
# =====================================================

if __name__ == "__main__":


    feature_engineering = FeatureEngineering()


    feature_engineering.initiate_feature_engineering()