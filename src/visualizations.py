"""
Visualizations Module
Generates academic-quality, publication-ready figures for dataset meta-features,
algorithm recommendation scores, model performance comparisons, and evaluation metrics.
All figures are automatically saved to outputs/ and returned for Streamlit display.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-blocking headless backend
import matplotlib.pyplot as plt
import seaborn as sns


def get_outputs_dir() -> str:
    """Returns absolute path to outputs directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# Configure global modern academic plot style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#D1D5DB",
    "axes.linewidth": 1.2,
    "grid.color": "#F3F4F6",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.autolayout": True
})


def plot_dataset_characteristics(meta_features: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Generates a visual summary infographic card of extracted dataset meta-features.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.axis("off")

    # Card background
    fig.patch.set_facecolor("#FFFFFF")
    
    # Title
    ax.text(0.02, 0.92, "Dataset Meta-Feature Profile", fontsize=16, fontweight="bold", color="#1E293B")
    ax.text(0.02, 0.84, "Summary of extracted structural and statistical meta-attributes", fontsize=10, color="#64748B")

    # Key statistics cards
    stats = [
        ("Total Instances (n)", f"{meta_features.get('number_of_rows', 0):,}", "#2563EB"),
        ("Feature Count (p)", f"{meta_features.get('number_of_features', 0)}", "#7C3AED"),
        ("Dataset Scale", f"{meta_features.get('dataset_size', 'N/A')}", "#059669"),
        ("Dimensionality", f"{meta_features.get('dimensionality', 'N/A')}", "#D97706"),
        ("Target Paradigm", f"{meta_features.get('target_type', 'N/A').capitalize()}", "#DC2626"),
        ("Missing Values", f"{meta_features.get('missing_values_percentage', 0.0)}%", "#4F46E5"),
    ]

    from matplotlib.patches import FancyBboxPatch, Rectangle

    # Draw metric tiles
    col_width = 0.30
    row_height = 0.32
    for i, (label, val, accent_color) in enumerate(stats):
        r = i // 3
        c = i % 3
        x = 0.02 + c * (col_width + 0.03)
        y = 0.44 - r * (row_height + 0.04)

        # Bounding box
        rect = FancyBboxPatch((x, y), col_width, row_height,
                              boxstyle="round,pad=0.02,rounding_size=0.03",
                              facecolor="#F8FAFC", edgecolor="#E2E8F0",
                              transform=ax.transAxes, linewidth=1.2)
        ax.add_patch(rect)

        # Left color bar
        bar = Rectangle((x, y), 0.008, row_height, facecolor=accent_color, edgecolor="none",
                        transform=ax.transAxes)
        ax.add_patch(bar)

        # Labels inside tile
        ax.text(x + 0.02, y + row_height - 0.09, label, fontsize=9, fontweight="bold", color="#64748B", transform=ax.transAxes)
        ax.text(x + 0.02, y + 0.07, val, fontsize=14, fontweight="bold", color="#0F172A", transform=ax.transAxes)

    if save_path is None:
        save_path = os.path.join(get_outputs_dir(), "dataset_characteristics.png")
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    return fig


def plot_missing_values(meta_features: Dict[str, Any], save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plots a bar chart of missing value percentages across columns.
    """
    missing_by_col = meta_features.get("missing_by_column", {})
    if not missing_by_col:
        return None

    # Filter columns with missing values or display top columns
    cols = []
    pcts = []
    counts = []

    for col, data in missing_by_col.items():
        cols.append(col)
        pcts.append(data.get("missing_percentage", 0.0))
        counts.append(data.get("missing_count", 0))

    df_miss = pd.DataFrame({"Feature": cols, "MissingPct": pcts, "MissingCount": counts})
    df_miss = df_miss.sort_values(by="MissingPct", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    
    # If all 0 missing values
    if df_miss["MissingPct"].max() == 0:
        ax.text(0.5, 0.5, "No Missing Values Detected in Dataset (100% Complete)",
                horizontalalignment="center", verticalalignment="center",
                fontsize=13, fontweight="bold", color="#059669", transform=ax.transAxes)
        ax.axis("off")
    else:
        palette = sns.color_palette("Reds_r", n_colors=len(df_miss))
        bars = ax.barh(df_miss["Feature"], df_miss["MissingPct"], color=palette, edgecolor="#991B1B", height=0.6)
        
        # Add labels
        for bar, count in zip(bars, df_miss["MissingCount"]):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{width:.1f}% ({count} nulls)", va="center", fontsize=8.5, color="#1E293B")

        ax.set_xlabel("Missing Percentage (%)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_title("Missing Value Distribution by Feature", fontsize=12, fontweight="bold", color="#1E293B", pad=12)
        ax.set_xlim(0, max(df_miss["MissingPct"].max() * 1.25, 10))
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    if save_path is None:
        save_path = os.path.join(get_outputs_dir(), "missing_values.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return fig


def plot_feature_types(meta_features: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots a modern donut chart of numerical vs categorical feature counts.
    """
    n_num = meta_features.get("numerical_features_count", 0)
    n_cat = meta_features.get("categorical_features_count", 0)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)

    labels = ["Numerical Features", "Categorical Features"]
    values = [n_num, n_cat]
    colors = ["#2563EB", "#F59E0B"]

    # Filter non-zero for clean pie
    valid_data = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not valid_data:
        valid_data = [("Features", 1, "#64748B")]

    plot_labels, plot_vals, plot_colors = zip(*valid_data)

    wedges, texts, autotexts = ax.pie(
        plot_vals,
        labels=plot_labels,
        colors=plot_colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 9, "fontweight": "bold", "color": "#1E293B"},
        wedgeprops={"width": 0.45, "edgecolor": "#FFFFFF", "linewidth": 2}
    )

    for autotext in autotexts:
        autotext.set_color("#FFFFFF")
        autotext.set_fontsize(10)

    ax.set_title(f"Feature Modality Breakdown (Total Features: {n_num + n_cat})", fontsize=11, fontweight="bold", pad=12)

    if save_path is None:
        save_path = os.path.join(get_outputs_dir(), "feature_types.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return fig


def plot_recommendation_scores(recommendations: List[Dict[str, Any]], save_path: Optional[str] = None) -> plt.Figure:
    """
    Generates a horizontal bar chart displaying Strategy Router algorithm recommendation scores.
    """
    algorithms = [r["algorithm"] for r in recommendations]
    scores = [r["score"] for r in recommendations]

    # Invert order for top algorithm at top of bar chart
    algorithms = algorithms[::-1]
    scores = scores[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    
    # Gradient palette from blue to emerald
    colors = ["#059669" if i == len(scores) - 1 else "#3B82F6" for i in range(len(scores))]

    bars = ax.barh(algorithms, scores, color=colors, edgecolor="#1E293B", height=0.55, linewidth=0.8)

    # Annotate score values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f} / 10.0", va="center", fontsize=9, fontweight="bold", color="#0F172A")

    ax.set_xlabel("Meta-Learning Recommendation Score (0-10)", fontsize=10, fontweight="bold", color="#334155")
    ax.set_title("Strategy Router: Candidate Algorithm Fitness Scores", fontsize=12, fontweight="bold", color="#0F172A", pad=14)
    ax.set_xlim(0, 11.5)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    if save_path is None:
        save_path = os.path.join(get_outputs_dir(), "recommendation_chart.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return fig


def plot_model_comparison(evaluation_output: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots a multi-metric comparison of the trained candidate models.
    """
    target_type = evaluation_output.get("target_type", "classification")
    models = evaluation_output.get("evaluated_models", [])
    valid_models = [m for m in models if m.get("status") == "SUCCESS"]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)

    if not valid_models:
        ax.text(0.5, 0.5, "No successfully evaluated models to compare.", ha="center", va="center")
        ax.axis("off")
        return fig

    algos = [m["algorithm"] for m in valid_models]
    x = np.arange(len(algos))
    width = 0.28

    if target_type == "classification":
        accs = [m["metrics"]["accuracy"] for m in valid_models]
        f1s = [m["metrics"]["f1_score"] for m in valid_models]
        cv_means = [m["metrics"]["cv_mean"] for m in valid_models]

        rects1 = ax.bar(x - width, accs, width, label="Test Accuracy", color="#2563EB", edgecolor="#1E3A8A")
        rects2 = ax.bar(x, f1s, width, label="Test F1-Score (Weighted)", color="#059669", edgecolor="#065F46")
        rects3 = ax.bar(x + width, cv_means, width, label="5-Fold CV Mean", color="#7C3AED", edgecolor="#5B21B6")

        ax.set_ylabel("Metric Score (0 - 1.0)", fontsize=10, fontweight="bold")
        ax.set_title("Candidate Models Performance Comparison (Classification)", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylim(0, 1.15)

    else:
        r2s = [m["metrics"]["r2_score"] for m in valid_models]
        rmses = [m["metrics"]["rmse"] for m in valid_models]
        cv_r2 = [m["metrics"]["cv_mean"] for m in valid_models]

        rects1 = ax.bar(x - width/2, r2s, width, label="Test R² Score", color="#2563EB", edgecolor="#1E3A8A")
        rects2 = ax.bar(x + width/2, cv_r2, width, label="5-Fold CV R² Mean", color="#7C3AED", edgecolor="#5B21B6")

        ax.set_ylabel("R² Coefficient of Determination", fontsize=10, fontweight="bold")
        ax.set_title("Candidate Models Performance Comparison (Regression)", fontsize=12, fontweight="bold", pad=12)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=9, fontweight="bold", rotation=10)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if save_path is None:
        save_path = os.path.join(get_outputs_dir(), "model_comparison.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return fig


def plot_confusion_matrix(best_model_info: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Renders an annotated confusion matrix heatmap using actual test predictions of the best model.
    """
    metrics = best_model_info.get("metrics", {})
    cm = np.array(metrics.get("confusion_matrix", [[1, 0], [0, 1]]))
    labels = metrics.get("class_labels", [str(i) for i in range(cm.shape[0])])
    algo_name = best_model_info.get("algorithm", "Best Model")

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=200)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=1.2,
        linecolor="#F1F5F9",
        annot_kws={"size": 11, "fontweight": "bold"},
        ax=ax
    )

    ax.set_title(f"Confusion Matrix — {algo_name}", fontsize=12, fontweight="bold", color="#0F172A", pad=14)
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylabel("True Actual Class", fontsize=10, fontweight="bold", color="#334155")

    # Add accuracy note
    acc = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1_score", 0.0)
    plt.figtext(0.5, -0.04, f"Test Accuracy: {acc:.4f}  |  Weighted F1-Score: {f1:.4f}",
                ha="center", fontsize=9.5, fontweight="bold", color="#1E293B")

    if save_path is None:
        save_path = os.path.join(get_outputs_dir(), "confusion_matrix.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return fig


def plot_actual_vs_predicted(best_model_info: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Generates Actual vs Predicted scatter plot with perfect-prediction 45-degree line for regression.
    """
    metrics = best_model_info.get("metrics", {})
    y_true = np.array(metrics.get("y_test_values", []))
    y_pred = np.array(metrics.get("y_pred_values", []))
    algo_name = best_model_info.get("algorithm", "Best Regressor")

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=200)

    if len(y_true) > 0 and len(y_pred) > 0:
        ax.scatter(y_true, y_pred, color="#2563EB", alpha=0.6, edgecolors="none", s=35, label="Test Samples")
        
        # 45-degree identity reference line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="#DC2626", linestyle="--", linewidth=1.5, label="Ideal Fit (y = x)")

        ax.set_xlabel("Actual Target Values", fontsize=10, fontweight="bold", color="#334155")
        ax.set_ylabel("Predicted Target Values", fontsize=10, fontweight="bold", color="#334155")
        ax.set_title(f"Actual vs Predicted — {algo_name}", fontsize=12, fontweight="bold", color="#0F172A", pad=12)

        r2 = metrics.get("r2_score", 0.0)
        rmse = metrics.get("rmse", 0.0)
        plt.figtext(0.5, -0.04, f"Test R²: {r2:.4f}  |  RMSE: {rmse:.4f}",
                    ha="center", fontsize=9.5, fontweight="bold", color="#1E293B")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

    if save_path is None:
        save_path = os.path.join(get_outputs_dir(), "actual_vs_predicted.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return fig


def generate_all_visualizations(
    meta_features: Dict[str, Any],
    router_results: Dict[str, Any],
    evaluation_results: Dict[str, Any]
) -> Dict[str, str]:
    """
    Batch helper that produces and saves all academic charts to outputs/.
    """
    out_paths = {}

    # 1. Dataset characteristics
    path1 = os.path.join(get_outputs_dir(), "dataset_characteristics.png")
    plot_dataset_characteristics(meta_features, save_path=path1)
    out_paths["dataset_characteristics"] = path1

    # 2. Missing values
    path2 = os.path.join(get_outputs_dir(), "missing_values.png")
    plot_missing_values(meta_features, save_path=path2)
    out_paths["missing_values"] = path2

    # 3. Feature types
    path3 = os.path.join(get_outputs_dir(), "feature_types.png")
    plot_feature_types(meta_features, save_path=path3)
    out_paths["feature_types"] = path3

    # 4. Recommendation scores
    path4 = os.path.join(get_outputs_dir(), "recommendation_chart.png")
    plot_recommendation_scores(router_results.get("all_ranked_algorithms", []), save_path=path4)
    out_paths["recommendation_chart"] = path4

    # 5. Model comparison
    path5 = os.path.join(get_outputs_dir(), "model_comparison.png")
    plot_model_comparison(evaluation_results, save_path=path5)
    out_paths["model_comparison"] = path5

    # 6. Confusion matrix or Actual vs Predicted
    best_model = evaluation_results.get("best_model")
    if best_model and evaluation_results.get("target_type") == "classification":
        path6 = os.path.join(get_outputs_dir(), "confusion_matrix.png")
        plot_confusion_matrix(best_model, save_path=path6)
        out_paths["confusion_matrix"] = path6
    elif best_model and evaluation_results.get("target_type") == "regression":
        path7 = os.path.join(get_outputs_dir(), "actual_vs_predicted.png")
        plot_actual_vs_predicted(best_model, save_path=path7)
        out_paths["actual_vs_predicted"] = path7

    return out_paths
