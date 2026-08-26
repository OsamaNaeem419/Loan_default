import os
import sys
import json
import pickle

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.models.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src import config


class TrainPipeline:
    """
    End-to-end training for the two-stage system's Stage-2 model.
    Saves four artifacts: model.pkl, preprocessor.pkl, columns.pkl and
    threshold.json — everything the serving pipeline needs.
    """

    def initiate_training_pipeline(self):
        try:
            logging.info("Training pipeline started")

            train_path, val_path, test_path = DataIngestion().initiate_data_ingestion()

            (X_train, y_train, X_val, y_val, X_test, y_test,
             preprocessor, model_columns) = (
                DataTransformation().initiate_data_transformation(
                    train_path, val_path, test_path)
            )

            os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
            with open(config.COLUMNS_PATH, "wb") as f:
                pickle.dump(model_columns, f)
            logging.info("Model column order saved")

            model_path, threshold = ModelTrainer().initiate_model_training(
                X_train, y_train, X_val, y_val, X_test, y_test)

            with open(config.THRESHOLD_PATH, "w") as f:
                json.dump({"threshold": threshold}, f, indent=2)

            logging.info(f"Training pipeline completed. Threshold={threshold:.4f}")

            print("\n==============================")
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("==============================")
            print(f"Model:      {model_path}")
            print(f"Preprocessor: {config.PREPROCESSOR_PATH}")
            print(f"Columns:    {config.COLUMNS_PATH}")
            print(f"Threshold:  {threshold:.4f} -> {config.THRESHOLD_PATH}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainPipeline().initiate_training_pipeline()
