from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.models.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys
import pickle
import os


class TrainPipeline:
    def initiate_training_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # -----------------------------
            # STEP 1: DATA INGESTION
            # -----------------------------
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # -----------------------------
            # STEP 2: DATA TRANSFORMATION
            # -----------------------------
            transformation = DataTransformation()
            X_train, X_test, y_train, y_test, preprocessor, columns = transformation.initiate_data_transformation(
                train_path, test_path
            )

            # -----------------------------
            # SAVE COLUMN ORDER (CRITICAL FIX)
            # -----------------------------
            os.makedirs("artifacts", exist_ok=True)

            with open("artifacts/columns.pkl", "wb") as f:
                pickle.dump(columns, f)

            logging.info("Column order saved successfully")

            # -----------------------------
            # STEP 3: MODEL TRAINING
            # -----------------------------
            trainer = ModelTrainer()
            model_path = trainer.initiate_model_training(
                X_train, X_test, y_train, y_test
            )

            logging.info(f"Training pipeline completed. Model saved at {model_path}")

            print("\n==============================")
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("==============================")
            print("Model saved at:", model_path)

        except Exception as e:
            raise CustomException(e, sys)


# -----------------------------
# RUN PIPELINE
# -----------------------------
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.initiate_training_pipeline()