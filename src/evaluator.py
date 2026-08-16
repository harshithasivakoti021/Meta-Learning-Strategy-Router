"""
Evaluator Module
Computes standard academic performance metrics, generates confusion matrices for classification,
and residual/scatter metrics for regression tasks.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)


def evaluate_trained_models(training_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates all successfully trained candidate models and selects the top performer.

    Parameters:
        training_output (dict): Output from model_trainer.train_and_cross_validate_candidates

    Returns:
        evaluation_results (dict): Structured metrics, confusion matrices, and best model metadata.
    """
    target_type = training_output.get("target_type", "classification")
    trained_models = training_output.get("trained_models", [])

    evaluated_models = []

    for model_info in trained_models:
        if model_info.get("status") != "SUCCESS":
            evaluated_models.append(model_info)
            continue

        y_test = model_info["y_test"]
        y_pred = model_info["y_pred"]
        algorithm = model_info["algorithm"]

        if target_type == "classification":
            # Classification Metrics
            acc = float(accuracy_score(y_test, y_pred))
            
            # Use weighted & macro averaging for multi-class and imbalanced data
            prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

            # Compute confusion matrix
            classes = sorted(list(pd.Series(y_test).unique()))
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            # Normalized confusion matrix (percentage per actual row)
            cm_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")

            metrics = {
                "accuracy": round(acc, 4),
                "precision": round(prec_weighted, 4),
                "recall": round(rec_weighted, 4),
                "f1_score": round(f1_weighted, 4),
                "precision_macro": round(prec_macro, 4),
                "recall_macro": round(rec_macro, 4),
                "f1_score_macro": round(f1_macro, 4),
                "cv_mean": model_info.get("cv_mean", 0.0),
                "cv_std": model_info.get("cv_std", 0.0),
                "training_time": model_info.get("training_time", 0.0),
                "confusion_matrix": cm.tolist(),
                "confusion_matrix_normalized": cm_norm.tolist(),
                "class_labels": [str(c) for c in classes]
            }

        else:
            # Regression Metrics
            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))
            residuals = (y_test - y_pred).tolist()

            metrics = {
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "r2_score": round(r2, 4),
                "cv_mean": model_info.get("cv_mean", 0.0),
                "cv_std": model_info.get("cv_std", 0.0),
                "training_time": model_info.get("training_time", 0.0),
                "residuals": residuals,
                "y_test_values": [float(v) for v in y_test],
                "y_pred_values": [float(v) for v in y_pred]
            }

        model_eval_record = {
            **model_info,
            "metrics": metrics
        }
        evaluated_models.append(model_eval_record)

    # Determine Best Model
    successful_models = [m for m in evaluated_models if m.get("status") == "SUCCESS"]
    best_model = None

    if successful_models:
        if target_type == "classification":
            # Primary ranking metric: test F1-score (or CV mean as tie-breaker)
            best_model = max(
                successful_models,
                key=lambda m: (m["metrics"]["f1_score"], m["metrics"]["accuracy"], m["metrics"]["cv_mean"])
            )
        else:
            # Primary ranking metric: test R2 score (or lowest RMSE)
            best_model = max(
                successful_models,
                key=lambda m: (m["metrics"]["r2_score"], -m["metrics"]["rmse"])
            )

    return {
        "target_type": target_type,
        "evaluated_models": evaluated_models,
        "best_model": best_model,
        "total_evaluated": len(successful_models)
    }
