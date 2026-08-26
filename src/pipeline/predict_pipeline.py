import sys
import json
import pickle
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.pipeline.explain import PredictionExplainer
from src import config


class PredictPipeline:
    """
    Serves the two-stage decision.

    Stage 1 — a prior default on file is an automatic rejection, decided in
    code with a plain-language reason and no model call.

    Stage 2 — everyone else is scored by the XGBoost model, thresholded, and
    explained with SHAP.
    """

    def __init__(self):
        self.model = pickle.load(open(config.MODEL_PATH, "rb"))
        self.preprocessor = pickle.load(open(config.PREPROCESSOR_PATH, "rb"))
        self.columns = pickle.load(open(config.COLUMNS_PATH, "rb"))
        with open(config.THRESHOLD_PATH) as f:
            self.threshold = json.load(f)["threshold"]
        self.explainer = PredictionExplainer(self.model, self.preprocessor, self.columns)

    def predict(self, input_data):
        try:
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()

            # ---- STAGE 1: policy gate ---------------------------------------
            gate_value = input_df.iloc[0].get(config.POLICY_COLUMN, config.POLICY_REJECT_VALUE)
            if gate_value == config.POLICY_REJECT_VALUE:
                logging.info("Stage 1: rejected by policy gate (prior default on file)")
                return {
                    "prediction": 0,
                    "probability": 0.0,
                    "stage": "policy_gate",
                    "reason": config.POLICY_REASON,
                    "explanation": None,
                    "counterfactual": None,
                }

            # ---- STAGE 2: model ---------------------------------------------
            X = input_df.reindex(columns=self.columns, fill_value=0)
            proba, pred, transformed = self._score(X)

            return {
                "prediction": int(pred),
                "probability": float(proba),
                "stage": "model",
                "reason": None,
                "explanation": self.explainer.explain(transformed, X),
                "counterfactual": self._policy_counterfactual(),
            }

        except Exception as e:
            raise CustomException(e, sys)

    def _score(self, X):
        transformed = self.preprocessor.transform(X)
        proba = float(self.model.predict_proba(transformed)[:, 1][0])
        pred = int(proba >= self.threshold)
        return proba, pred, transformed

    @staticmethod
    def _policy_counterfactual():
        # This applicant passed Stage 1. State plainly what a prior default
        # would have meant — a deterministic rejection, no model involved.
        return {
            "field": config.POLICY_COLUMN,
            "note": ("A prior default on file would trigger an automatic "
                     "rejection at Stage 1, regardless of every other detail."),
            "prediction": 0,
        }
