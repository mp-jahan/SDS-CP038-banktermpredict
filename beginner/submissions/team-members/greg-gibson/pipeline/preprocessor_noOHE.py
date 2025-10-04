import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# -----------------------------
# Custom transformers
# -----------------------------
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        drop_cols = [c for c in self.columns if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
        return X


class YesNoEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].map({"yes": 1, "no": 0}).astype("Int64")
        return X


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, caps):
        self.caps = caps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.caps.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X


class PdaysTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "pdays" in X.columns:
            X["never_contacted"] = (X["pdays"] == -1).astype(int)
            X["pdays"] = X["pdays"].replace(-1, 0)
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Avoid division by zero via (previous + 1)
        X["campaign_per_previous"] = X["campaign"] / (X["previous"] + 1)
        # housing is expected to be 0/1 after YesNoEncoder
        X["balance_x_housing"] = X["balance"] * X["housing"]
        # split days into sine and cosine
        X["day_sin"] = np.sin(X["day"] * (2 * np.pi / 30))
        X["day_cos"] = np.cos(X["day"] * (2 * np.pi / 30))
        return X


class AgeBinner(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    Bin age into categories and return a DataFrame with column 'age_group'.
    Works with sklearn's set_output(transform="pandas").
    """
    def __init__(self, bins=None, labels=None):
        self.bins = bins or [0, 25, 40, 60, 100]
        self.labels = labels or ["young", "adult", "mature", "senior"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            age = X.iloc[:, 0] if X.shape[1] == 1 else X["age"]
        else:
            age = pd.Series(X.ravel())
        age_group = pd.cut(age, bins=self.bins, labels=self.labels, right=False)
        return pd.DataFrame({"age_group": age_group}, index=age.index)


# Add a CategoryCaster transformer for LightGBM/CatBoost compatibility
class CategoryCaster(BaseEstimator, TransformerMixin):
    """
    Cast selected columns to Pandas 'category' dtype
    (for LightGBM/CatBoost compatibility).
    """
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X
    

# -----------------------------
# Keep categorical, no OHE preprocessor
# -----------------------------
def noOHE_preprocessor():
    """
    Smaller preprocessing pipeline for LightGBM/CatBoost that performs:
    - Drop leakage column: duration
    - Yes/No -> Binary for: default, housing, loan
    - Outlier capping for: age, balance, campaign, previous
    - pdays handling: create never_contacted flag (pdays == -1) and set -1 -> 0
    - Feature engineering: campaign_per_previous, balance_x_housing, day_sin, day_cos
    - Ordinal encoding for education (unknown:0, primary:1, secondary:2, tertiary:3)
    - Age binning -> age groups [young, adult, mature, senior]
    """

    yes_no_cols = ["default", "housing", "loan"]

    outlier_caps = {
        "age": (18, 100),
        "balance": (-10000, 40000),
        "campaign": (1, 30),
        "previous": (0, 20),
    }

    # Categorical encoders
    education_ordinal = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(categories=[["unknown", "primary", "secondary", "tertiary"]]))
    ])

    # Column-wise transformer for age binning, updated to accommodate .set_output(transform="pandas").
    age_binning = Pipeline(steps=[
        ("age_binner", AgeBinner()),
    ]).set_output(transform="pandas")

    columnwise = ColumnTransformer(
        transformers=[
            ("education", education_ordinal, ["education"]),
            ("age_binned", age_binning, ["age"]),
            ("categoricals", "passthrough", ["job", "marital", "contact", "month", "poutcome"]),           
        ],
        remainder="drop",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    

    pipeline = Pipeline(steps=[
        ("yesno", YesNoEncoder(columns=yes_no_cols)),
        ("cap", OutlierCapper(caps=outlier_caps)),
        ("pdays", PdaysTransformer()),
        ("engineer", FeatureEngineer()),
        ("drop", ColumnDropper(columns=["duration", "day"])),
        ("columns", columnwise),
        ("cat_cast", CategoryCaster(categorical_cols=["age_group", "job", "marital", "contact", "month", "poutcome"])),
    ])

    return pipeline
