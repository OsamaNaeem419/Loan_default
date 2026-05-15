import os
import sys
import pickle
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from src.logger import logging
from src.exception import CustomException


class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting model training")

            # -----------------------------
            # Handle imbalance
            # -----------------------------
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

            # -----------------------------
            # Model
            # -----------------------------
            model = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42
            )

            # -----------------------------
            # Train
            # -----------------------------
            model.fit(X_train, y_train)

            # -----------------------------
            # Predict probabilities
            # -----------------------------
            y_proba = model.predict_proba(X_test)[:, 1]

            threshold = 0.88
            y_pred = (y_proba >= threshold).astype(int)

            # -----------------------------
            # Evaluation
            # -----------------------------
            roc = roc_auc_score(y_test, y_proba)

            logging.info(f"ROC-AUC: {roc}")

            print("\nROC-AUC:", roc)
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # -----------------------------
            # Save model
            # -----------------------------
            os.makedirs("artifacts", exist_ok=True)

            pickle.dump(model, open(self.model_path, "wb"))

            logging.info(f"Model saved at {self.model_path}")

            return self.model_path

        except Exception as e:
            raise CustomException(e, sys)