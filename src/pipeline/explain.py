import sys

import numpy as np
import scipy.sparse as sp
import shap

from src.exception import CustomException
from src.logger import logging


# Human-readable names for the raw model columns, so the API can hand the UI
# something presentable instead of `cb_person_cred_hist_length`.
FEATURE_LABELS = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_emp_exp": "Work experience",
    "person_home_ownership": "Home ownership",
    "loan_amnt": "Loan amount",
    "loan_intent": "Loan intent",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan % of income",
    "cb_person_cred_hist_length": "Credit history length",
    "credit_score": "Credit score",
    "previous_loan_defaults_on_file": "Previous defaults",
}

# Contributions smaller than this (in log-odds) are rounding noise, not signal.
MIN_IMPACT = 1e-3


class PredictionExplainer:
    """
    Explains a single prediction with TreeSHAP.

    The model is trained on the one-hot expanded matrix (20 columns), but a user
    thinks in terms of the 11 fields they actually filled in. SHAP values are
    additive, so the contributions of the dummy columns belonging to one
    categorical field can simply be summed back into that field.
    """

    def __init__(self, model, preprocessor, source_columns):
        self.explainer = shap.TreeExplainer(model)
        self.source_columns = list(source_columns)
        self.owner_of_column = self._map_transformed_to_source(
            preprocessor, self.source_columns
        )

    @staticmethod
    def _map_transformed_to_source(preprocessor, source_columns):
        """
        Map each transformed column back to the raw column it came from, e.g.
        `cat__person_home_ownership_RENT` -> `person_home_ownership`
        `remainder__credit_score`         -> `credit_score`
        """
        owners = []

        for name in preprocessor.get_feature_names_out():
            # Strip the ColumnTransformer's `<transformer>__` prefix.
            bare = name.split("__", 1)[1] if "__" in name else name

            if bare in source_columns:
                owners.append(bare)
                continue

            # One-hot column: `<source_column>_<category>`. Raw column names
            # contain underscores themselves, so match on the longest source
            # column that prefixes this name rather than splitting on "_".
            candidates = [c for c in source_columns if bare.startswith(c + "_")]

            if candidates:
                owners.append(max(candidates, key=len))
            else:
                logging.warning(f"Could not map transformed feature '{name}'")
                owners.append(bare)

        return owners

    def explain(self, transformed_row, raw_row):
        """
        Returns the per-field contributions for one applicant.

        `transformed_row` is the preprocessor output, `raw_row` the one-row
        DataFrame of values the applicant submitted. Contributions are in
        log-odds: positive pushes toward approval, negative away from it.
        """
        try:
            dense = (
                transformed_row.toarray()
                if sp.issparse(transformed_row)
                else np.asarray(transformed_row)
            )

            shap_values = np.asarray(self.explainer.shap_values(dense))[0]

            # Fold the one-hot columns back into their source field.
            totals = dict.fromkeys(self.source_columns, 0.0)
            for owner, value in zip(self.owner_of_column, shap_values):
                totals[owner] = totals.get(owner, 0.0) + float(value)

            submitted = raw_row.iloc[0].to_dict()
            scale = sum(abs(v) for v in totals.values()) or 1.0

            contributions = [
                {
                    "feature": column,
                    "label": FEATURE_LABELS.get(column, column),
                    "value": self._plain(submitted.get(column)),
                    # Log-odds contribution: signed, additive, model-space.
                    "impact": round(impact, 4),
                    # Share of this prediction's total movement, for bar widths.
                    "weight": round(abs(impact) / scale, 4),
                }
                for column, impact in totals.items()
                if abs(impact) >= MIN_IMPACT
            ]

            contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)

            return {
                "supported": [c for c in contributions if c["impact"] > 0],
                "underwhelmed": [c for c in contributions if c["impact"] < 0],
            }

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _plain(value):
        """Convert numpy scalars to JSON-serialisable Python types."""
        if isinstance(value, np.generic):
            return value.item()
        return value
