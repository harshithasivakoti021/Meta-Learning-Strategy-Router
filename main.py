"""
Meta-Learning Machine Learning Strategy Router - Main CLI Runner
Demonstrates the full end-to-end dataset analysis, algorithm scoring, model training,
metric evaluation, and visualization generation from the command line.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

from src.dataset_loader import load_openml_dataset, load_csv_dataset
from src.dataset_analyzer import analyze_dataset
from src.strategy_router import StrategyRouter
from src.model_trainer import train_and_cross_validate_candidates
from src.evaluator import evaluate_trained_models
from src.visualizations import generate_all_visualizations, get_outputs_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StrategyRouterCLI")


def run_pipeline(
    dataset_source: str,
    target_column: str = None,
    max_samples: int = 5000,
    top_k: int = 3,
    save_json: bool = True
) -> Dict[str, Any]:
    """
    Executes the full Meta-Learning Strategy Router pipeline.
    """
    print("=" * 80)
    print(" META-LEARNING MACHINE LEARNING STRATEGY ROUTER ".center(80, "="))
    print(" An Explainable Dataset-Aware Algorithm Recommendation System ".center(80, " "))
    print("=" * 80)

    # 1. Dataset Loading
    logger.info(f"Loading dataset source: '{dataset_source}'...")
    if dataset_source.endswith(".csv") and os.path.exists(dataset_source):
        df, default_target, metadata = load_csv_dataset(dataset_source, max_samples=max_samples)
    else:
        df, default_target, metadata = load_openml_dataset(dataset_source, max_samples=max_samples)

    target_col = target_column or default_target
    logger.info(f"Dataset '{metadata.get('dataset_name')}' loaded successfully: {len(df)} rows, {len(df.columns)} columns.")
    logger.info(f"Target column: '{target_col}'")

    # 2. Meta-Feature Analysis
    logger.info("Extracting dataset meta-features...")
    meta_features = analyze_dataset(df, target_column=target_col)

    print("\n" + "-" * 40 + " META-FEATURES " + "-" * 40)
    print(f" Instances (n)             : {meta_features['number_of_rows']}")
    print(f" Features (p)              : {meta_features['number_of_features']} ({meta_features['numerical_features_count']} numeric, {meta_features['categorical_features_count']} categorical)")
    print(f" Missing Values            : {meta_features['missing_values_count']} ({meta_features['missing_values_percentage']}%)")
    print(f" Target Type               : {meta_features['target_type'].upper()} ({meta_features['target_details'].get('sub_type', 'N/A')})")
    print(f" Dataset Size Category     : {meta_features['dataset_size']}")
    print(f" Dimensionality Category   : {meta_features['dimensionality']} (p/n ratio: {meta_features['feature_to_sample_ratio']})")
    print("-" * 95)

    # 3. Strategy Routing
    logger.info("Executing Strategy Router algorithm scoring...")
    router = StrategyRouter()
    routing_result = router.route_and_recommend(meta_features, top_k=top_k)

    print("\n" + "-" * 35 + " ALGORITHM RECOMMENDATIONS " + "-" * 35)
    for rec in routing_result["all_ranked_algorithms"]:
        tag = " [TOP RECOMMENDATION]" if rec["rank"] == 1 else ""
        print(f"\nRank {rec['rank']}: {rec['algorithm']} - Score: {rec['score']}/10.0{tag}")
        print(f"  * Reasons:")
        for r in rec["reasons"]:
            print(f"    - {r}")
        if rec["warnings"]:
            print(f"  * Caveats / Warnings:")
            for w in rec["warnings"]:
                print(f"    ! {w}")
        print(f"  * Preprocessing: {rec['preprocessing_requirements']}")
    print("-" * 95)

    print("\n" + "-" * 35 + " ROUTER EXPLANATION " + "-" * 35)
    print(routing_result["holistic_explanation"])
    print("-" * 95)

    # 4. Model Training & Cross-Validation
    logger.info(f"Training top {len(routing_result['top_recommendations'])} candidate models with 5-Fold Cross-Validation...")
    training_output = train_and_cross_validate_candidates(
        df=df,
        target_column=target_col,
        meta_features=meta_features,
        top_candidates=routing_result["top_recommendations"],
        test_size=0.2,
        cv_folds=5
    )

    # 5. Model Evaluation
    logger.info("Evaluating trained models and computing performance metrics...")
    evaluation_result = evaluate_trained_models(training_output)

    print("\n" + "-" * 35 + " MODEL EVALUATION RESULTS " + "-" * 35)
    if evaluation_result["target_type"] == "classification":
        print(f"{'Rank':<5} | {'Algorithm':<30} | {'CV Mean':<8} | {'Test Acc':<8} | {'F1-Score':<8} | {'Time (s)':<8}")
        print("-" * 80)
        for m in evaluation_result["evaluated_models"]:
            if m.get("status") == "SUCCESS":
                met = m["metrics"]
                print(f"{m['router_rank']:<5} | {m['algorithm']:<30} | {met['cv_mean']:<8.4f} | {met['accuracy']:<8.4f} | {met['f1_score']:<8.4f} | {met['training_time']:<8.4f}")
            else:
                print(f"{m['router_rank']:<5} | {m['algorithm']:<30} | FAILED ({m.get('error_message')})")
    else:
        print(f"{'Rank':<5} | {'Algorithm':<30} | {'CV R²':<8} | {'Test R²':<8} | {'RMSE':<8} | {'Time (s)':<8}")
        print("-" * 80)
        for m in evaluation_result["evaluated_models"]:
            if m.get("status") == "SUCCESS":
                met = m["metrics"]
                print(f"{m['router_rank']:<5} | {m['algorithm']:<30} | {met['cv_mean']:<8.4f} | {met['r2_score']:<8.4f} | {met['rmse']:<8.4f} | {met['training_time']:<8.4f}")
    print("-" * 80)

    best_m = evaluation_result.get("best_model")
    if best_m:
        print(f"\n>>> BEST PERFORMING MODEL: {best_m['algorithm']}")
        if evaluation_result["target_type"] == "classification":
            print(f"    Test Accuracy: {best_m['metrics']['accuracy']:.4f}  |  Weighted F1: {best_m['metrics']['f1_score']:.4f}  |  5-Fold CV: {best_m['metrics']['cv_mean']:.4f}")
        else:
            print(f"    Test R²: {best_m['metrics']['r2_score']:.4f}  |  RMSE: {best_m['metrics']['rmse']:.4f}  |  5-Fold CV R²: {best_m['metrics']['cv_mean']:.4f}")

    # 6. Generate and Save Visualizations
    logger.info("Generating and saving academic visualization charts to outputs/...")
    chart_paths = generate_all_visualizations(meta_features, routing_result, evaluation_result)
    for chart_name, path in chart_paths.items():
        print(f"  [SAVED] {os.path.basename(path)}")

    # 7. Save JSON Summary Report
    if save_json:
        report_data = {
            "dataset_metadata": metadata,
            "meta_features": {
                k: v for k, v in meta_features.items()
                if k not in ["missing_by_column"]  # omit detailed column breakdown for clean json
            },
            "recommendations": routing_result["all_ranked_algorithms"],
            "holistic_explanation": routing_result["holistic_explanation"],
            "model_evaluations": [
                {
                    "algorithm": m["algorithm"],
                    "router_rank": m["router_rank"],
                    "router_score": m["router_score"],
                    "training_time": m.get("metrics", {}).get("training_time", 0.0),
                    "metrics": {k: v for k, v in m.get("metrics", {}).items() if k not in ["residuals", "y_test_values", "y_pred_values"]}
                }
                for m in evaluation_result["evaluated_models"] if m.get("status") == "SUCCESS"
            ],
            "best_model": {
                "algorithm": best_m["algorithm"] if best_m else "None",
                "metrics": {k: v for k, v in best_m.get("metrics", {}).items() if k not in ["residuals", "y_test_values", "y_pred_values"]} if best_m else {}
            }
        }
        json_path = os.path.join(get_outputs_dir(), "router_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"  [SAVED] router_results.json")

    print("\n" + "=" * 80)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY ".center(80, "="))
    print("=" * 80 + "\n")

    return {
        "meta_features": meta_features,
        "routing_result": routing_result,
        "evaluation_result": evaluation_result,
        "chart_paths": chart_paths
    }


def main():
    parser = argparse.ArgumentParser(
        description="Meta-Learning Machine Learning Strategy Router CLI Runner"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Iris",
        help="Dataset name ('Titanic', 'Iris', 'Breast Cancer', 'Diabetes', 'MNIST'), OpenML ID, or path to local CSV."
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target column name. If omitted, heuristically auto-detected."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum sample rows to load for interactive evaluation (default: 5000)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top recommended models to train and evaluate (default: 3)."
    )

    args = parser.parse_args()
    run_pipeline(
        dataset_source=args.dataset,
        target_column=args.target,
        max_samples=args.max_samples,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
