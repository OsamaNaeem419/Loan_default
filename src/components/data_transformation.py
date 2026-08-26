import os
import sys
import pickle
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from src import config


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = config.PREPROCESSOR_PATH


class DataTransformation:
    """
    Applies the Stage-1 gate to each split, drops the non-model columns, and
    one-hot encodes the categoricals. The model is trained only on gated
    applicants (no prior default) — the exact population it will score in
    production (notebook Section 10).
    """

    def __init__(self):
        self.cfg = DataTransformationConfig()

    def _gate_and_xy(self, df):
        # Stage-1 gate: keep only applicants without a prior default.
        gated = df[df[config.POLICY_COLUMN] != config.POLICY_REJECT_VALUE]
        features = config.model_features(df.columns.tolist())
        return gated[features], gated[config.TARGET]

    def initiate_data_transformation(self, train_path, val_path, test_path):
        try:
            logging.info("Data transformation started (gate + encode)")

            X_train, y_train = self._gate_and_xy(pd.read_csv(train_path))
            X_val, y_val = self._gate_and_xy(pd.read_csv(val_path))
            X_test, y_test = self._gate_and_xy(pd.read_csv(test_path))

            model_columns = X_train.columns.tolist()
            categorical_cols = X_train.select_dtypes(
                include=["object", "string"]).columns.tolist()

            logging.info(f"Model features ({len(model_columns)}): {model_columns}")
            logging.info(f"Categorical: {categorical_cols}")
            logging.info(f"Gated rows — train {len(X_train):,} / "
                         f"val {len(X_val):,} / test {len(X_test):,}")

            # Tree model: one-hot the categoricals, pass numerics through as-is.
            preprocessor = ColumnTransformer(
                transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),
                               categorical_cols)],
                remainder="passthrough",
            )

            X_train_t = preprocessor.fit_transform(X_train)
            X_val_t = preprocessor.transform(X_val)
            X_test_t = preprocessor.transform(X_test)

            os.makedirs(os.path.dirname(self.cfg.preprocessor_path), exist_ok=True)
            with open(self.cfg.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)
            logging.info("Preprocessor saved")

            return (X_train_t, y_train, X_val_t, y_val, X_test_t, y_test,
                    preprocessor, model_columns)

        except Exception as e:
            raise CustomException(e, sys)
