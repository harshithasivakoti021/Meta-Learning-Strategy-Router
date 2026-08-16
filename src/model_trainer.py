"""
Model Trainer Module
Instantiates, trains, and cross-validates recommended candidate machine learning pipelines
with timing, stratification, and leak-free execution.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

from src.preprocessing import create_model_pipeline

logger = logging.getLogger(__name__)


def instantiate_estimator(model_key: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Factory creating Scikit-Learn estimators from router keys with default/recommended hyperparameters.
    """
    params = params.copy() if params else {}

    # Classification Estimators
    if model_key == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    elif model_key == "GradientBoostingClassifier":
        return GradientBoostingClassifier(**params)
    elif model_key == "LogisticRegression":
        return LogisticRegression(**params)
    elif model_key == "SVC":
        return SVC(**params)
    elif model_key == "KNeighborsClassifier":
        return KNeighborsClassifier(**params)
    elif model_key == "DecisionTreeClassifier":
        return DecisionTreeClassifier(**params)
    elif model_key == "GaussianNB":
        return GaussianNB(**params)

    # Regression Estimators
    elif model_key == "GradientBoostingRegressor":
        return GradientBoostingRegressor(**params)
    elif model_key == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    elif model_key == "Ridge":
        return Ridge(**params)
    elif model_key == "LinearRegression":
        return LinearRegression(**params)
    elif model_key == "DecisionTreeRegressor":
        return DecisionTreeRegressor(**params)
    elif model_key == "KNeighborsRegressor":
        return KNeighborsRegressor(**params)

    else:
        raise ValueError(f"Unrecognized model key: {model_key}")


def train_and_cross_validate_candidates(
    df: pd.DataFrame,
    target_column: str,
    meta_features: Dict[str, Any],
    top_candidates: List[Dict[str, Any]],
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains and evaluates top recommended algorithm pipelines.

    Returns:
        results (dict): Training and evaluation metrics for each candidate.
    """
    # 1. Clean missing target rows if any
    clean_df = df.dropna(subset=[target_column]).copy()
    feature_cols = [c for c in clean_df.columns if c != target_column]
    
    X = clean_df[feature_cols]
    y = clean_df[target_column]

    target_type = meta_features.get("target_type", "classification")
    numerical_features = meta_features.get("numerical_features", [])
    categorical_features = meta_features.get("categorical_features", [])

    # Filter feature lists to only existing columns
    numerical_features = [c for c in numerical_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]

    # Convert object/string y to categorical codes if classification to ensure consistency
    class_mapping = None
    if target_type == "classification":
        # Preserve original class names
        unique_classes = np.sort(pd.Series(y).unique())
        class_mapping = {val: idx for idx, val in enumerate(unique_classes)}
        # Keep y as numpy array or series
        stratify_y = y
        # Check if any class has fewer instances than cv_folds
        val_counts = pd.Series(y).value_counts()
        min_class_count = val_counts.min()
        if min_class_count < cv_folds:
            logger.warning(f"Class with only {min_class_count} instances detected. Falling back to non-stratified CV.")
            stratify_split = None
            cv_splitter = KFold(n_splits=min(cv_folds, max(min_class_count, 2)), shuffle=True, random_state=random_state)
        else:
            stratify_split = stratify_y
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        stratify_split = None
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # 2. Train-Test Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_split
        )
    except Exception as split_err:
        logger.warning(f"Stratified split failed ({split_err}). Using unstratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    trained_models = []

    # 3. Train Each Recommended Candidate Pipeline
    for candidate in top_candidates:
        model_name = candidate["algorithm"]
        model_key = candidate["model_key"]
        params = candidate.get("recommended_params", {})
        requires_scaling = candidate.get("requires_scaling", True)
        router_score = candidate.get("score", 0.0)
        router_rank = candidate.get("rank", 1)

        logger.info(f"Training Candidate {router_rank}: {model_name}...")

        try:
            estimator = instantiate_estimator(model_key, params)
            pipeline = create_model_pipeline(
                model_estimator=estimator,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                include_scaling=requires_scaling
            )

            # Measure Cross-Validation Score
            scoring_metric = "accuracy" if target_type == "classification" else "r2"
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring_metric)
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            # Measure Full Training Time on X_train
            start_time = time.perf_counter()
            pipeline.fit(X_train, y_train)
            training_time = round(time.perf_counter() - start_time, 4)

            # Predict on Test Set
            y_pred = pipeline.predict(X_test)
            y_pred_proba = None
            if target_type == "classification" and hasattr(pipeline, "predict_proba"):
                try:
                    y_pred_proba = pipeline.predict_proba(X_test)
                except Exception:
                    y_pred_proba = None

            trained_models.append({
                "algorithm": model_name,
                "model_key": model_key,
                "router_score": router_score,
                "router_rank": router_rank,
                "pipeline": pipeline,
                "training_time": training_time,
                "cv_scores": [float(s) for s in cv_scores],
                "cv_mean": round(cv_mean, 4),
                "cv_std": round(cv_std, 4),
                "y_test": y_test,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "reasons": candidate.get("reasons", []),
                "warnings": candidate.get("warnings", []),
                "requires_scaling": requires_scaling,
                "status": "SUCCESS"
            })

        except Exception as train_err:
            logger.error(f"Failed to train {model_name}: {train_err}")
            trained_models.append({
                "algorithm": model_name,
                "model_key": model_key,
                "router_score": router_score,
                "router_rank": router_rank,
                "status": "FAILED",
                "error_message": str(train_err)
            })

    return {
        "target_type": target_type,
        "target_column": target_column,
        "class_mapping": class_mapping,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "trained_models": trained_models
    }
