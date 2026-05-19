import os
import sys
import pandas as pd
import numpy as np
import pickle

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self, categorical_cols):
        try:
            logging.info("Creating preprocessing object")

            categorical_transformer = OneHotEncoder(handle_unknown="ignore")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", categorical_transformer, categorical_cols)
                ],
                remainder="passthrough"
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "loan_status"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # 🔥 SAVE COLUMN ORDER BEFORE TRANSFORM
            columns = X_train.columns.tolist()

            categorical_cols = X_train.select_dtypes(include=["object"]).columns
            logging.info(f"Categorical columns: {list(categorical_cols)}")

            preprocessor = self.get_data_transformer_object(categorical_cols)

            # FIT + TRANSFORM
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation completed")

            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path),
                        exist_ok=True)

            pickle.dump(preprocessor,
                        open(self.config.preprocessor_obj_file_path, "wb"))

            logging.info("Preprocessor saved successfully")

            # 🔥 RETURN COLUMNS TOO (CRITICAL FIX)
            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.config.preprocessor_obj_file_path,
                columns
            )

        except Exception as e:
            raise CustomException(e, sys)