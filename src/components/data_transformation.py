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


# ----------------------------
# CONFIG CLASS
# ----------------------------
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


# ----------------------------
# DATA TRANSFORMATION CLASS
# ----------------------------
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self, categorical_cols):
        """
        This function creates the preprocessing object (OneHotEncoder pipeline)
        """
        try:
            logging.info("Creating preprocessing object (OneHotEncoder)")

            # OneHotEncoder
            categorical_transformer = OneHotEncoder(
                handle_unknown="ignore"
            )

            # Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", categorical_transformer, categorical_cols)
                ],
                remainder="passthrough"  # keep numerical columns as is
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            # ----------------------------
            # LOAD DATA
            # ----------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            # ----------------------------
            # SPLIT X AND Y
            # ----------------------------
            target_column = "loan_status"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # ----------------------------
            # IDENTIFY CATEGORICAL COLUMNS
            # (edit based on your dataset)
            # ----------------------------
            categorical_cols = X_train.select_dtypes(include=["object"]).columns

            logging.info(f"Categorical columns: {list(categorical_cols)}")

            # ----------------------------
            # GET PREPROCESSOR OBJECT
            # ----------------------------
            preprocessor = self.get_data_transformer_object(categorical_cols)

            # ----------------------------
            # FIT ON TRAIN, TRANSFORM BOTH
            # ----------------------------
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation completed")

            # ----------------------------
            # SAVE PREPROCESSOR
            # ----------------------------
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path),
                        exist_ok=True)

            pickle.dump(preprocessor,
                        open(self.config.preprocessor_obj_file_path, "wb"))

            logging.info(f"Preprocessor saved at {self.config.preprocessor_obj_file_path}")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
        
