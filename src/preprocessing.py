"""
Preprocessing Pipeline Module
Builds leakage-free Scikit-Learn Pipelines and ColumnTransformers tailored dynamically
to candidate algorithm requirements (e.g., selective feature scaling, categorical encoding,
and missing value imputation).
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    include_scaling: bool = True,
    scale_method: str = "standard"
) -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer for mixed feature processing.

    Parameters:
        numerical_features (list): Names of continuous/numerical columns.
        categorical_features (list): Names of categorical/discrete columns.
        include_scaling (bool): Whether to apply feature scaling (StandardScaler/RobustScaler).
                               Set to False for tree-based models to avoid unnecessary compute.
        scale_method (str): "standard" for StandardScaler or "robust" for RobustScaler.

    Returns:
        preprocessor (ColumnTransformer): Configured transformer ready for pipeline embedding.
    """
    transformers = []

    # 1. Numerical Pipeline
    if numerical_features and len(numerical_features) > 0:
        num_steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if include_scaling:
            scaler = RobustScaler() if scale_method == "robust" else StandardScaler()
            num_steps.append(("scaler", scaler))

        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipeline, numerical_features))

    # 2. Categorical Pipeline
    if categorical_features and len(categorical_features) > 0:
        # Check compatibility with sparse_output (scikit-learn >= 1.2) vs sparse
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder)
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    # If no features defined at all
    if not transformers:
        return ColumnTransformer(transformers=[("identity", "passthrough", [])], remainder="passthrough")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # drop unhandled columns to prevent leakage
        verbose_feature_names_out=False
    )
    return preprocessor


def create_model_pipeline(
    model_estimator: Any,
    numerical_features: List[str],
    categorical_features: List[str],
    include_scaling: bool = True
) -> Pipeline:
    """
    Wraps preprocessor and estimator into a unified Scikit-Learn Pipeline.
    Ensures strict zero-leakage during cross-validation and testing.
    """
    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        include_scaling=include_scaling
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_estimator)
    ])

    return pipeline
