"""
Dataset Loader Module
Provides unified loading for OpenML datasets, local CSV datasets, and uploaded CSV files
with automatic offline fallbacks and caching.
"""

import os
import io
import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Dataset registry mapping names to OpenML IDs and metadata
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Titanic": {
        "openml_id": 40945,
        "default_target": "survived",
        "fallback_filename": "titanic.csv",
        "description": "Passenger survival binary classification dataset with mixed feature types and missing values.",
        "type": "classification"
    },
    "Iris": {
        "openml_id": 61,
        "default_target": "class",
        "fallback_filename": "iris.csv",
        "description": "Classic multiclass biological dataset with purely numerical measurements and no missing values.",
        "type": "classification"
    },
    "Breast Cancer": {
        "openml_id": 13,
        "default_target": "Class",
        "fallback_filename": "breast_cancer.csv",
        "description": "Diagnostic medical binary classification dataset with categorical/ordinal attributes and missing values.",
        "type": "classification"
    },
    "Diabetes": {
        "openml_id": 37,
        "default_target": "class",
        "fallback_filename": "diabetes.csv",
        "description": "Pima Indians diabetes binary classification dataset with numerical physiological metrics.",
        "type": "classification"
    },
    "MNIST": {
        "openml_id": 554,
        "default_target": "class",
        "fallback_filename": "mnist.csv",
        "description": "High-dimensional handwritten digits (784 pixel features, 10 classes) benchmark dataset.",
        "type": "classification"
    }
}


def get_dataset_dir() -> str:
    """Returns absolute path to the local datasets/ directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


def load_openml_dataset(
    dataset_identifier: Any,
    max_samples: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Optional[str], Dict[str, Any]]:
    """
    Attempts to fetch dataset from OpenML by ID or registered name.
    Falls back to local CSV if network or API fails.

    Returns:
        df (pd.DataFrame): Dataset dataframe
        default_target (str or None): Suggested target column
        metadata (dict): Metadata including source, ID, full size, etc.
    """
    import socket
    socket.setdefaulttimeout(5.0)

    dataset_name = "Custom OpenML Dataset"
    default_target = None
    openml_id = None
    fallback_file = None

    # Check if dataset_identifier is in our registry
    if isinstance(dataset_identifier, str) and dataset_identifier in DATASET_REGISTRY:
        reg = DATASET_REGISTRY[dataset_identifier]
        dataset_name = dataset_identifier
        openml_id = reg["openml_id"]
        default_target = reg["default_target"]
        fallback_file = reg["fallback_filename"]
    else:
        try:
            openml_id = int(dataset_identifier)
            dataset_name = f"OpenML Dataset #{openml_id}"
            # Check if any registered item matches this ID
            for name, reg in DATASET_REGISTRY.items():
                if reg["openml_id"] == openml_id:
                    dataset_name = name
                    default_target = reg["default_target"]
                    fallback_file = reg["fallback_filename"]
                    break
        except ValueError:
            pass

    # 1. If local benchmark file is present in datasets/, load directly for zero-latency execution
    if fallback_file:
        local_path = os.path.join(get_dataset_dir(), fallback_file)
        if os.path.exists(local_path):
            logger.info(f"Loading benchmark dataset from local cache: {local_path}")
            df = pd.read_csv(local_path)
            total_rows = len(df)
            is_sampled = False

            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
                is_sampled = True

            target_col = default_target
            if target_col and target_col not in df.columns:
                for col in df.columns:
                    if col.lower() == target_col.lower():
                        target_col = col
                        break

            metadata = {
                "dataset_name": dataset_name,
                "openml_id": openml_id,
                "source": f"OpenML Benchmark Dataset (ID {openml_id}) - Local Cache",
                "total_rows": total_rows,
                "loaded_rows": len(df),
                "is_sampled": is_sampled,
                "default_target": target_col or df.columns[-1],
                "description": DATASET_REGISTRY.get(dataset_name, {}).get("description", "Benchmark dataset.")
            }
            return df, metadata["default_target"], metadata

    # 2. Fetch from OpenML API for custom dataset IDs or missing files
    if openml_id is not None:
        try:
            import openml
            logger.info(f"Attempting to fetch dataset ID {openml_id} from OpenML...")
            dataset = openml.datasets.get_dataset(openml_id, download_data=True)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe"
            )
            
            target_col = dataset.default_target_attribute or default_target or "target"
            
            if y is not None:
                if isinstance(y, pd.Series):
                    df = X.copy()
                    df[target_col] = y
                elif isinstance(y, pd.DataFrame):
                    df = X.copy()
                    for col in y.columns:
                        df[col] = y[col]
                    target_col = y.columns[0]
                else:
                    df = X.copy()
                    df[target_col] = y
            else:
                df = X.copy()

            total_rows = len(df)
            is_sampled = False

            if fallback_file:
                local_path = os.path.join(get_dataset_dir(), fallback_file)
                try:
                    df.to_csv(local_path, index=False)
                    logger.info(f"Successfully cached {dataset_name} to {local_path}")
                except Exception as save_err:
                    logger.warning(f"Could not cache dataset locally: {save_err}")

            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
                is_sampled = True

            metadata = {
                "dataset_name": dataset_name,
                "openml_id": openml_id,
                "source": "OpenML API (Live)",
                "total_rows": total_rows,
                "loaded_rows": len(df),
                "is_sampled": is_sampled,
                "default_target": target_col,
                "description": dataset.description if hasattr(dataset, "description") else ""
            }
            return df, target_col, metadata

        except Exception as e:
            logger.warning(f"OpenML API fetch failed for ID {openml_id} ({e}). Switching to synthetic generator...")

    # 3. Fallback: generate built-in standard dataset
    df, target_col = _generate_fallback_dataset(dataset_name)
    total_rows = len(df)
    is_sampled = False
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
        is_sampled = True

    metadata = {
        "dataset_name": dataset_name,
        "openml_id": openml_id,
        "source": "Standard Scikit-Learn Generator (Offline Fallback)",
        "total_rows": total_rows,
        "loaded_rows": len(df),
        "is_sampled": is_sampled,
        "default_target": target_col,
        "description": f"Standard {dataset_name} reference dataset."
    }
    return df, target_col, metadata


def load_csv_dataset(
    file_source: Any,
    filename: Optional[str] = "uploaded_dataset.csv",
    max_samples: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Optional[str], Dict[str, Any]]:
    """
    Loads dataset from a CSV filepath, buffer, or Streamlit UploadedFile object.
    """
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise FileNotFoundError(f"CSV file not found at: {file_source}")
        df = pd.read_csv(file_source)
        source_name = os.path.basename(file_source)
    elif hasattr(file_source, "read"):
        # File buffer or Streamlit UploadedFile
        df = pd.read_csv(file_source)
        source_name = getattr(file_source, "name", filename)
    elif isinstance(file_source, pd.DataFrame):
        df = file_source.copy()
        source_name = filename or "DataFrame"
    else:
        raise ValueError(f"Unsupported file source type: {type(file_source)}")

    total_rows = len(df)
    is_sampled = False
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
        is_sampled = True

    metadata = {
        "dataset_name": source_name,
        "openml_id": None,
        "source": f"User Upload / Custom CSV ({source_name})",
        "total_rows": total_rows,
        "loaded_rows": len(df),
        "is_sampled": is_sampled,
        "default_target": df.columns[-1] if len(df.columns) > 0 else None,
        "description": "User-provided custom dataset."
    }
    return df, metadata["default_target"], metadata


def _generate_fallback_dataset(name: str) -> Tuple[pd.DataFrame, str]:
    """Generates standard sklearn datasets if neither OpenML nor local CSV are found."""
    from sklearn import datasets

    if name.lower() == "iris":
        data = datasets.load_iris(as_frame=True)
        df = data.frame.copy()
        target_name = "target"
        # Map target numbers to species names for realistic categorical representation
        target_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        df["species"] = df["target"].map(target_map)
        df = df.drop(columns=["target"])
        return df, "species"

    elif name.lower() in ["breast cancer", "breast_cancer"]:
        data = datasets.load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        target_map = {0: "malignant", 1: "benign"}
        df["diagnosis"] = df["target"].map(target_map)
        df = df.drop(columns=["target"])
        return df, "diagnosis"

    elif name.lower() == "diabetes":
        data = datasets.load_diabetes(as_frame=True)
        df = data.frame.copy()
        # In sklearn diabetes is regression; in OpenML 37 it is Pima diabetes classification
        return df, "target"

    elif name.lower() == "titanic":
        # Realistic Titanic synthetic distribution fallback
        np.random.seed(42)
        n = 891
        pclass = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
        sex = np.random.choice(["male", "female"], size=n, p=[0.65, 0.35])
        age = np.random.normal(29.7, 14.5, size=n).clip(1, 80)
        # Introduce realistic missing values
        age[np.random.choice(n, int(n * 0.2), replace=False)] = np.nan
        sibsp = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.68, 0.23, 0.04, 0.03, 0.02])
        parch = np.random.choice([0, 1, 2], size=n, p=[0.76, 0.13, 0.11])
        fare = (pclass * -20 + 80 + np.random.exponential(15, size=n)).clip(5, 500)
        embarked = np.random.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])
        
        # Survival probabilities
        p_surv = 0.38 + (sex == "female") * 0.4 - (pclass - 1) * 0.15 - (age > 60) * 0.1
        p_surv = np.clip(p_surv, 0.05, 0.95)
        survived = (np.random.rand(n) < p_surv).astype(int)

        df = pd.DataFrame({
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked,
            "survived": survived
        })
        return df, "survived"

    elif name.lower() == "mnist":
        data = datasets.load_digits(as_frame=True)
        df = data.frame.copy()
        return df, "target"

    else:
        # Generic synthetic dataset
        X, y = datasets.make_classification(n_samples=500, n_features=10, n_informative=6, random_state=42)
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
        df["target"] = y
        return df, "target"
