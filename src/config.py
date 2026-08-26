"""
Central configuration for the two-stage loan approval system.

Keeping the policy rule, the excluded columns, the feature list and the
artifact paths in one place means the training pipeline and the serving
pipeline can never quietly disagree about what the model expects.

Design reference: notebooks/loan_approval_system.ipynb (Sections 7-11).
"""
import os

RANDOM_STATE = 42
TARGET = "loan_status"

# ---------------------------------------------------------------------------
# Stage 1 — policy gate (a deterministic "knockout" rule, not a model input).
# In this dataset, previous_loan_defaults_on_file == "Yes" rejects with zero
# exceptions across 22,776 applicants, so it is executed as policy and the
# model never sees a prior defaulter. (Notebook Sections 7-9.)
# ---------------------------------------------------------------------------
POLICY_COLUMN = "previous_loan_defaults_on_file"
POLICY_REJECT_VALUE = "Yes"
POLICY_REASON = "prior_default_on_file"

# ---------------------------------------------------------------------------
# Dropped from the model entirely: no measurable signal, and person_gender is
# also a protected attribute that must not drive a credit decision. (Sec 8.2-8.3)
# ---------------------------------------------------------------------------
DROP_COLUMNS = ["person_gender", "person_education"]

# Data quality: ages above this are corrupted records (Section 3.3).
MAX_PLAUSIBLE_AGE = 90

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "raw", "loan_data.csv")
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "columns.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.json")


def model_features(all_columns):
    """The columns the Stage-2 model is trained and served on: everything
    except the policy column, the dropped columns, and the target."""
    excluded = set(DROP_COLUMNS + [POLICY_COLUMN, TARGET])
    return [c for c in all_columns if c not in excluded]
