import pandas as pd
import numpy as np
import os
import pickle
import sys

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        self.model = pickle.load(open("artifacts/model.pkl", "rb"))
        self.preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))
        self.columns = pickle.load(open("artifacts/columns.pkl", "rb"))  # 🔥 IMPORTANT
        self.threshold = 0.88

    def predict(self, input_data):
        try:
            logging.info("Prediction started")

            # -----------------------------
            # Convert input to DataFrame
            # -----------------------------
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data

            # -----------------------------
            # 🔥 FORCE COLUMN ALIGNMENT
            # -----------------------------
            input_df = input_df.reindex(columns=self.columns, fill_value=0)

            # -----------------------------
            # Transform
            # -----------------------------
            transformed = self.preprocessor.transform(input_df)

            # -----------------------------
            # Predict
            # -----------------------------
            proba = self.model.predict_proba(transformed)[:, 1]
            pred = (proba >= self.threshold).astype(int)

            return {
                "prediction": int(pred[0]),
                "probability": float(proba[0])
            }

        except Exception as e:
            raise CustomException(e, sys)