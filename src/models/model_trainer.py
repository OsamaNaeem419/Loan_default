import os
import sys
import pickle
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve,
)

from src.logger import logging
from src.exception import CustomException
from src import config


class ModelTrainer:
    """
    Trains the Stage-2 XGBoost model on the gated training set, selects the
    decision threshold on the gated validation set, and reports on the gated
    test set. Threshold and reporting use different splits on purpose, so the
    reported numbers stay honest (notebook Section 11).
    """

    def __init__(self):
        self.model_path = config.MODEL_PATH

    @staticmethod
    def _max_f1_threshold(y_val, val_proba):
        # Threshold that maximises F1 on the validation set. This is a
        # PLACEHOLDER: it treats a false approval and a false rejection as
        # equally costly, which they are not. Re-derive from real loan
        # economics before production (notebook Section 11.4).
        prec, rec, thr = precision_recall_curve(y_val, val_proba)
        f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
        return float(thr[int(np.argmax(f1s))])

    def initiate_model_training(self, X_train, y_train, X_val, y_val, X_test, y_test):
        try:
            logging.info("Stage-2 model training started")

            # On the gated population the imbalance is mild, so scale_pos_weight
            # sits near 1 — but computing it keeps the code correct if the data
            # changes.
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            model = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=config.RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            # Threshold on validation, report on test.
            val_proba = model.predict_proba(X_val)[:, 1]
            threshold = self._max_f1_threshold(y_val, val_proba)

            test_proba = model.predict_proba(X_test)[:, 1]
            test_pred = (test_proba >= threshold).astype(int)
            roc = roc_auc_score(y_test, test_proba)

            logging.info(f"scale_pos_weight={scale_pos_weight:.3f} "
                         f"threshold={threshold:.4f} test_roc_auc={roc:.4f}")

            print(f"\nscale_pos_weight:                 {scale_pos_weight:.3f}")
            print(f"Threshold (max-F1 on validation): {threshold:.4f}   [placeholder — see notebook 11.4]")
            print(f"Test ROC-AUC:                     {roc:.4f}")
            print("\nConfusion matrix (test):")
            print(confusion_matrix(y_test, test_pred))
            print("\nClassification report (test):")
            print(classification_report(y_test, test_pred,
                                        target_names=["rejected", "approved"]))

            os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)
            logging.info(f"Model saved at {self.model_path}")

            return self.model_path, float(threshold)

        except Exception as e:
            raise CustomException(e, sys)
