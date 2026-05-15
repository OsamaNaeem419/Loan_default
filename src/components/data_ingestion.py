import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.mlproject.utils import read_sql_data


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

            # STEP 1: READ FROM MYSQL
            df = read_sql_data()

            # STEP 2: CREATE ARTIFACT FOLDER
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # STEP 3: SAVE RAW DATA
            df.to_csv(self.config.raw_data_path, index=False)

            # STEP 4: TRAIN TEST SPLIT
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)