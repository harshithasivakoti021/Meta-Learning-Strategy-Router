"""
Dataset Analyzer Module
Extracts comprehensive statistical and structural meta-features from tabular datasets
to feed into the Meta-Learning Strategy Router.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def suggest_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically identifies a potential target column in a DataFrame.
    Looks for common convention names or defaults to the last column.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return None

    common_target_names = [
        "target", "label", "class", "survived", "outcome", "diagnosis",
        "y", "response", "dependent", "result", "status", "churn",
        "price", "salary", "medv", "sales", "revenue"
    ]
    
    # 1. Exact case-insensitive match
    for col in df.columns:
        if str(col).strip().lower() in common_target_names:
            return col

    # 2. Substring match
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if any(t in col_lower for t in ["target", "label", "class", "outcome", "churn"]):
            return col

    # 3. Default to the last column
    return df.columns[-1]


def detect_target_type(target_series: pd.Series) -> Tuple[str, Dict[str, Any]]:
    """
    Determines whether a target column represents a classification or regression problem,
    along with detailed target properties.
    """
    clean_target = target_series.dropna()
    total_valid = len(clean_target)
    
    if total_valid == 0:
        return "unknown", {"reason": "Target series has no non-null values."}

    unique_vals = clean_target.unique()
    n_unique = len(unique_vals)
    dtype = clean_target.dtype

    # Boolean or categorical/object dtypes are classification
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
        is_binary = (n_unique == 2)
        counts = clean_target.value_counts().to_dict()
        total = sum(counts.values())
        distribution = {str(k): {"count": int(v), "percentage": round(float(v) / total * 100, 2)} for k, v in counts.items()}
        
        # Calculate imbalance ratio (majority / minority)
        counts_list = list(counts.values())
        imbalance_ratio = round(max(counts_list) / max(min(counts_list), 1), 2)
        
        return "classification", {
            "sub_type": "binary" if is_binary else "multiclass",
            "number_of_classes": n_unique,
            "classes": [str(c) for c in unique_vals[:20]],
            "class_distribution": distribution,
            "class_imbalance_ratio": imbalance_ratio,
            "is_imbalanced": imbalance_ratio >= 2.0
        }

    # For numeric dtypes: check unique count and ratio
    # If distinct values are few (e.g. <= 15 or <= 5% of dataset), it's likely classification/discrete labels
    if n_unique <= 10 or (n_unique <= 20 and n_unique / total_valid < 0.05):
        # Treat as classification
        is_binary = (n_unique == 2)
        counts = clean_target.value_counts().to_dict()
        total = sum(counts.values())
        distribution = {str(k): {"count": int(v), "percentage": round(float(v) / total * 100, 2)} for k, v in counts.items()}
        counts_list = list(counts.values())
        imbalance_ratio = round(max(counts_list) / max(min(counts_list), 1), 2)

        return "classification", {
            "sub_type": "binary" if is_binary else "multiclass",
            "number_of_classes": n_unique,
            "classes": [str(c) for c in unique_vals[:20]],
            "class_distribution": distribution,
            "class_imbalance_ratio": imbalance_ratio,
            "is_imbalanced": imbalance_ratio >= 2.0
        }
    else:
        # Continuous numeric target -> Regression
        stats = {
            "mean": round(float(clean_target.mean()), 4),
            "std": round(float(clean_target.std()), 4),
            "min": round(float(clean_target.min()), 4),
            "max": round(float(clean_target.max()), 4),
            "median": round(float(clean_target.median()), 4),
            "skewness": round(float(clean_target.skew()), 4) if hasattr(clean_target, "skew") else 0.0
        }
        return "regression", {
            "sub_type": "continuous",
            "number_of_classes": None,
            "target_statistics": stats
        }


def analyze_dataset(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs full academic meta-feature extraction on a given dataset.

    Parameters:
        df (pd.DataFrame): Raw dataset dataframe.
        target_column (str, optional): Target column to analyze. If None, automatically detected.

    Returns:
        meta_features (dict): Complete meta-feature dictionary.
    """
    if df is None or df.empty:
        raise ValueError("Cannot analyze an empty or None DataFrame.")

    # 1. Target Column Identification
    if target_column is None or target_column not in df.columns:
        target_column = suggest_target_column(df)
        is_target_auto_detected = True
    else:
        is_target_auto_detected = False

    # 2. Features Separation
    feature_columns = [col for col in df.columns if col != target_column]
    X_df = df[feature_columns] if feature_columns else pd.DataFrame()
    y_series = df[target_column] if target_column in df.columns else None

    # 3. Basic Dimensions
    number_of_rows = int(len(df))
    number_of_features = int(len(feature_columns))
    duplicate_rows = int(df.duplicated().sum())

    # 4. Feature Types Detection
    numerical_features: List[str] = []
    categorical_features: List[str] = []
    
    for col in feature_columns:
        dtype = X_df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            # Check if integer column has very low cardinality and might act as category
            numerical_features.append(col)
        else:
            categorical_features.append(col)

    num_count = len(numerical_features)
    cat_count = len(categorical_features)
    has_mixed_types = (num_count > 0 and cat_count > 0)

    # 5. Missing Values Analysis
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    total_cells = number_of_rows * len(df.columns) if len(df.columns) > 0 else 1
    missing_percentage = round(float(total_missing / total_cells) * 100, 2)
    has_missing_values = (total_missing > 0)

    missing_by_column: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        cnt = int(missing_counts[col])
        pct = round(float(cnt / max(number_of_rows, 1)) * 100, 2)
        missing_by_column[col] = {
            "missing_count": cnt,
            "missing_percentage": pct,
            "has_missing": cnt > 0
        }

    # 6. Target Type & Complexity Analysis
    if y_series is not None:
        target_type, target_meta = detect_target_type(y_series)
    else:
        target_type = "unknown"
        target_meta = {}

    # 7. Dataset Size Categorization
    # Small: rows < 1000, Medium: 1000 <= rows < 10000, Large: rows >= 10000
    if number_of_rows < 1000:
        dataset_size_cat = "Small"
    elif number_of_rows < 10000:
        dataset_size_cat = "Medium"
    else:
        dataset_size_cat = "Large"

    # 8. Dimensionality Categorization
    # Low: features < 20, Medium: 20 <= features < 100, High: features >= 100
    feature_to_sample_ratio = round(float(number_of_features / max(number_of_rows, 1)), 4)
    if number_of_features < 20:
        dimensionality_cat = "Low"
    elif number_of_features < 100:
        dimensionality_cat = "Medium"
    else:
        dimensionality_cat = "High"

    # High p/n ratio override
    if feature_to_sample_ratio >= 0.1 and dimensionality_cat == "Low":
        dimensionality_cat = "Medium (High p/n ratio)"

    # 9. Assembly of Meta-Features
    meta_features: Dict[str, Any] = {
        "number_of_rows": number_of_rows,
        "number_of_features": number_of_features,
        "total_columns": int(len(df.columns)),
        "numerical_features": numerical_features,
        "numerical_features_count": num_count,
        "categorical_features": categorical_features,
        "categorical_features_count": cat_count,
        "has_mixed_types": has_mixed_types,
        "missing_values_count": total_missing,
        "missing_values_percentage": missing_percentage,
        "has_missing_values": has_missing_values,
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_percentage": round(float(duplicate_rows / max(number_of_rows, 1)) * 100, 2),
        "target_column": target_column,
        "is_target_auto_detected": is_target_auto_detected,
        "target_type": target_type,
        "target_details": target_meta,
        "dataset_size": dataset_size_cat,
        "dimensionality": dimensionality_cat,
        "feature_to_sample_ratio": feature_to_sample_ratio
    }

    return meta_features
