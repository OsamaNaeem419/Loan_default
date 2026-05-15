import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import read_sql_data


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            # STEP 1: READ DATA
            df = read_sql_data()
            logging.info(f"Data shape before processing: {df.shape}")

            # STEP 2: DROP UNNECESSARY COLUMNS
            drop_cols = ["person_education", "person_gender"]
            df = df.drop(columns=drop_cols, errors="ignore")
            logging.info(f"Dropped columns: {drop_cols}")

            # STEP 3: DROP MISSING VALUES (NEW ADDITION)
            missing_before = df.isnull().sum().sum()
            logging.info(f"Total missing values before dropna: {missing_before}")
            df = df[df["person_age"] <= 90]

            df = df.dropna()

            missing_after = df.isnull().sum().sum()
            logging.info(f"Total missing values after dropna: {missing_after}")

            logging.info(f"Data shape after cleaning: {df.shape}")

            # STEP 4: CREATE ARTIFACT FOLDER
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # STEP 5: SAVE RAW CLEANED DATA
            df.to_csv(self.config.raw_data_path, index=False)

            # STEP 6: TRAIN TEST SPLIT
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["loan_status"] if "loan_status" in df.columns else None
            )

            logging.info("Train-test split completed")

            # STEP 7: SAVE SPLITS
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")
            print("\n===== TRAIN DATA HEAD =====")
            print(train_df.head())

            print("\n===== TEST DATA HEAD =====")
            print(test_df.head())

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)



