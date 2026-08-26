import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src import config


@dataclass
class DataIngestionConfig:
    raw_data_path: str = config.RAW_DATA_PATH
    train_data_path: str = os.path.join(config.ARTIFACTS_DIR, "train.csv")
    val_data_path: str = os.path.join(config.ARTIFACTS_DIR, "val.csv")
    test_data_path: str = os.path.join(config.ARTIFACTS_DIR, "test.csv")


class DataIngestion:
    """
    Reads the raw CSV, cleans it, and produces a 70/15/15 stratified split.

    The policy column and the dropped columns are deliberately KEPT in these
    CSVs. The Stage-1 gate and feature selection happen after the split, in
    data_transformation, so every split stays a clean subset of one stratified
    partition of the whole dataset (notebook Section 10.1).
    """

    def __init__(self):
        self.cfg = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info(f"Data ingestion started (source: {self.cfg.raw_data_path})")
            df = pd.read_csv(self.cfg.raw_data_path)
            logging.info(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

            # Clean: remove implausible ages (the corrupted rows), then any NaNs.
            before = len(df)
            df = df[df["person_age"] <= config.MAX_PLAUSIBLE_AGE]
            df = df.dropna().reset_index(drop=True)
            logging.info(f"Cleaned: {before:,} -> {len(df):,} rows")

            # 70 / 15 / 15 stratified split on the target.
            train_df, temp_df = train_test_split(
                df, test_size=0.30, random_state=config.RANDOM_STATE,
                stratify=df[config.TARGET],
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.50, random_state=config.RANDOM_STATE,
                stratify=temp_df[config.TARGET],
            )

            os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
            train_df.to_csv(self.cfg.train_data_path, index=False)
            val_df.to_csv(self.cfg.val_data_path, index=False)
            test_df.to_csv(self.cfg.test_data_path, index=False)
            logging.info(
                f"Saved splits — train {len(train_df):,} / "
                f"val {len(val_df):,} / test {len(test_df):,}"
            )

            return (self.cfg.train_data_path,
                    self.cfg.val_data_path,
                    self.cfg.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)
