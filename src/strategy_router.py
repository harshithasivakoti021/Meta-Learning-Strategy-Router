"""
Strategy Router Module
Core meta-learning recommendation engine that scores, ranks, and provides explainable
justifications for candidate machine learning algorithms based on dataset meta-features.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class StrategyRouter:
    """
    Explainable Rule-Based Meta-Learning Strategy Router.
    Analyzes dataset meta-features and computes quantitative fitness scores (0-10)
    for candidate algorithms with dynamic natural language reasoning.
    """

    def __init__(self):
        pass

    def route_and_recommend(
        self,
        meta_features: Dict[str, Any],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Main entry point for algorithm routing.
        
        Parameters:
            meta_features (dict): Extracted dataset meta-features.
            top_k (int): Number of top ranked algorithms to return.

        Returns:
            result (dict): Complete recommendation package with rankings, scores, and justifications.
        """
        target_type = meta_features.get("target_type", "classification")

        if target_type == "classification":
            recommendations = self._score_classification_algorithms(meta_features)
        elif target_type == "regression":
            recommendations = self._score_regression_algorithms(meta_features)
        else:
            # Fallback to classification
            recommendations = self._score_classification_algorithms(meta_features)

        # Sort recommendations by score descending
        sorted_recs = sorted(recommendations, key=lambda x: x["score"], reverse=True)
        
        # Assign ranks
        for idx, rec in enumerate(sorted_recs):
            rec["rank"] = idx + 1

        top_recommendations = sorted_recs[:top_k]
        holistic_explanation = self._generate_holistic_explanation(meta_features, sorted_recs[0])

        return {
            "target_type": target_type,
            "all_ranked_algorithms": sorted_recs,
            "top_recommendations": top_recommendations,
            "best_recommendation": sorted_recs[0] if sorted_recs else None,
            "holistic_explanation": holistic_explanation,
            "meta_summary": {
                "rows": meta_features.get("number_of_rows"),
                "features": meta_features.get("number_of_features"),
                "size_category": meta_features.get("dataset_size"),
                "dimensionality": meta_features.get("dimensionality"),
                "missing_pct": meta_features.get("missing_values_percentage"),
                "has_mixed_types": meta_features.get("has_mixed_types")
            }
        }

    def _score_classification_algorithms(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculates explainable scores for candidate classification algorithms."""
        n_rows = meta.get("number_of_rows", 100)
        n_feats = meta.get("number_of_features", 10)
        n_num = meta.get("numerical_features_count", 0)
        n_cat = meta.get("categorical_features_count", 0)
        has_mixed = meta.get("has_mixed_types", False)
        missing_pct = meta.get("missing_values_percentage", 0.0)
        target_meta = meta.get("target_details", {})
        sub_type = target_meta.get("sub_type", "binary")
        n_classes = target_meta.get("number_of_classes", 2)
        imbalance_ratio = target_meta.get("class_imbalance_ratio", 1.0)
        is_imbalanced = target_meta.get("is_imbalanced", False)
        p_to_n = meta.get("feature_to_sample_ratio", 0.01)

        candidates = []

        # ==========================================
        # 1. Random Forest Classifier
        # ==========================================
        rf_score = 6.0
        rf_reasons = []
        rf_warnings = []
        rf_params = {"n_estimators": 100, "random_state": 42}

        rf_reasons.append("Robust ensemble model capable of capturing complex non-linear decision boundaries.")
        rf_score += 1.2

        if has_mixed or n_cat > 0:
            rf_reasons.append(f"Excels at heterogeneous mixed feature spaces ({n_num} numeric, {n_cat} categorical).")
            rf_score += 1.2

        if n_rows >= 1000 and n_rows < 10000:
            rf_reasons.append("Ideal dataset size for bagging ensembles with low variance.")
            rf_score += 0.8
        elif n_rows >= 10000:
            rf_reasons.append("Scales well to large datasets through tree bagging.")
            rf_score += 0.6
        else:
            rf_reasons.append("Provides strong regularization against overfitting on smaller samples.")
            rf_score += 0.5

        if is_imbalanced:
            rf_reasons.append(f"Class imbalance detected (ratio {imbalance_ratio}:1). Can be counteracted via class_weight='balanced'.")
            rf_params["class_weight"] = "balanced"
            rf_score += 0.8

        if missing_pct > 0:
            rf_reasons.append(f"Tolerant to imputed features with {missing_pct}% missingness.")
            rf_score += 0.4

        if n_feats >= 100 or p_to_n > 0.1:
            rf_reasons.append("Random feature subspace selection mitigates high dimensionality.")
            rf_score += 0.6

        rf_score = min(round(rf_score, 1), 9.8)
        candidates.append({
            "algorithm": "Random Forest Classifier",
            "model_key": "RandomForestClassifier",
            "score": rf_score,
            "reasons": rf_reasons,
            "warnings": rf_warnings or ["Requires tuning of n_estimators and max_depth on very deep structures."],
            "preprocessing_requirements": "Median/Mode imputation + One-Hot Encoding (Feature scaling is NOT strictly necessary for tree splits).",
            "recommended_params": rf_params,
            "requires_scaling": False
        })

        # ==========================================
        # 2. Gradient Boosting Classifier
        # ==========================================
        gb_score = 6.0
        gb_reasons = []
        gb_warnings = []
        gb_params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}

        gb_reasons.append("Sequential boosting produces state-of-the-art predictive performance on structured tabular data.")
        gb_score += 1.5

        if n_rows >= 500 and n_rows < 20000:
            gb_reasons.append("Dataset volume provides sufficient samples for gradient descent optimization without excessive training latency.")
            gb_score += 1.0
        elif n_rows < 500:
            gb_warnings.append("On very small datasets, boosting may overfit if learning rate is too high or max_depth is unconstrained.")
            gb_score -= 0.6
        elif n_rows >= 20000:
            gb_warnings.append("Sequential tree fitting may incur noticeable training time on large datasets.")

        if has_mixed:
            gb_reasons.append("Effectively handles interactions between continuous and one-hot encoded categorical variables.")
            gb_score += 0.8

        if is_imbalanced:
            gb_reasons.append("Focuses residual learning on hard-to-classify minority samples.")
            gb_score += 0.5

        gb_score = min(round(gb_score, 1), 9.6)
        candidates.append({
            "algorithm": "Gradient Boosting Classifier",
            "model_key": "GradientBoostingClassifier",
            "score": gb_score,
            "reasons": gb_reasons,
            "warnings": gb_warnings or ["Sensitive to noisy outliers in target labels."],
            "preprocessing_requirements": "Median/Mode imputation + One-Hot Encoding.",
            "recommended_params": gb_params,
            "requires_scaling": False
        })

        # ==========================================
        # 3. Logistic Regression
        # ==========================================
        lr_score = 5.0
        lr_reasons = []
        lr_warnings = []
        lr_params = {"max_iter": 1000, "random_state": 42}

        if sub_type == "binary":
            lr_reasons.append("Direct, highly interpretable probabilistic log-odds modeling for binary classification.")
            lr_score += 1.2
        else:
            lr_reasons.append(f"Supports multiclass ({n_classes} classes) via multinomial cross-entropy formulation.")
            lr_score += 0.8

        if n_feats >= 100 or p_to_n > 0.1:
            lr_reasons.append("Linear decision boundaries with L2/L1 regularization generalize very well in high-dimensional feature spaces.")
            lr_score += 1.5
            lr_params["penalty"] = "l2"
            lr_params["C"] = 1.0

        if n_cat == 0 and n_num > 0:
            lr_reasons.append("Clean numerical feature space allows effective hyper-plane separation.")
            lr_score += 0.9

        if is_imbalanced:
            lr_reasons.append("Integrates class_weight='balanced' to adjust intercept for class distribution skew.")
            lr_params["class_weight"] = "balanced"
            lr_score += 0.7

        if n_rows < 1000:
            lr_reasons.append("Low parameter count prevents severe overfitting on small sample sizes.")
            lr_score += 0.8

        if has_mixed:
            lr_warnings.append("Cannot model non-linear feature interactions without manual polynomial expansion.")
            lr_score -= 0.5

        lr_score = min(max(round(lr_score, 1), 3.0), 9.2)
        candidates.append({
            "algorithm": "Logistic Regression",
            "model_key": "LogisticRegression",
            "score": lr_score,
            "reasons": lr_reasons,
            "warnings": lr_warnings or ["Assumes linear relationship between log-odds and input features."],
            "preprocessing_requirements": "Imputation + One-Hot Encoding + Mandatory StandardScaler.",
            "recommended_params": lr_params,
            "requires_scaling": True
        })

        # ==========================================
        # 4. Support Vector Machine (SVM)
        # ==========================================
        svm_score = 5.0
        svm_reasons = []
        svm_warnings = []
        svm_params = {"random_state": 42}

        if n_feats >= 50 or p_to_n > 0.05:
            svm_reasons.append("Maximum-margin hyperplane optimization is highly effective in high-dimensional spaces.")
            svm_score += 1.8
            svm_params["kernel"] = "linear"
        else:
            svm_params["kernel"] = "rbf"
            svm_reasons.append("RBF kernel maps features into infinite-dimensional space to separate non-linear clusters.")
            svm_score += 1.0

        if n_rows < 1000:
            svm_reasons.append("Small sample size suits SVM's support-vector-centric formulation without computational bottleneck.")
            svm_score += 1.4
        elif n_rows >= 10000:
            svm_warnings.append("SVM quadratic-to-cubic complexity O(n^2 to n^3) makes training on large datasets computationally heavy.")
            svm_score -= 2.2

        if n_cat == 0 and n_num > 0:
            svm_reasons.append("Continuous numeric features create smooth geometric margin boundaries.")
            svm_score += 0.8

        if is_imbalanced:
            svm_params["class_weight"] = "balanced"
            svm_reasons.append("Configured with class_weight='balanced' to penalize margin violations on minority class.")
            svm_score += 0.6

        svm_score = min(max(round(svm_score, 1), 2.5), 9.5)
        candidates.append({
            "algorithm": "Support Vector Machine (SVM)",
            "model_key": "SVC",
            "score": svm_score,
            "reasons": svm_reasons,
            "warnings": svm_warnings or ["Strictly requires normalized/standardized numerical features."],
            "preprocessing_requirements": "Imputation + One-Hot Encoding + Mandatory StandardScaler.",
            "recommended_params": svm_params,
            "requires_scaling": True
        })

        # ==========================================
        # 5. K-Nearest Neighbors (KNN)
        # ==========================================
        knn_score = 4.5
        knn_reasons = []
        knn_warnings = []
        knn_params = {"n_neighbors": 5}

        if n_rows < 1000 and n_feats < 20 and n_cat == 0:
            knn_reasons.append("Non-parametric local metric clustering provides intuitive boundaries on small low-dimensional data.")
            knn_score += 2.0
        elif n_rows < 2000 and n_feats < 30:
            knn_reasons.append("Reasonable sample and feature volume for metric distance computation.")
            knn_score += 0.8

        if n_feats >= 50 or p_to_n > 0.1:
            knn_warnings.append("Curse of dimensionality: in high dimensions, Euclidean distances converge to equal lengths.")
            knn_score -= 2.0

        if n_rows >= 10000:
            knn_warnings.append("Inference time scales linearly with dataset size O(N), causing slow query latency.")
            knn_score -= 1.8

        if has_mixed or n_cat > 0:
            knn_warnings.append("Euclidean distance loses physical geometric fidelity on sparse one-hot categorical vectors.")
            knn_score -= 1.0

        knn_score = min(max(round(knn_score, 1), 2.0), 8.5)
        candidates.append({
            "algorithm": "K-Nearest Neighbors (KNN)",
            "model_key": "KNeighborsClassifier",
            "score": knn_score,
            "reasons": knn_reasons or ["Simple non-parametric instance-based baseline."],
            "warnings": knn_warnings or ["Sensitive to feature scale and irrelevant attributes."],
            "preprocessing_requirements": "Imputation + One-Hot Encoding + Mandatory StandardScaler.",
            "recommended_params": knn_params,
            "requires_scaling": True
        })

        # ==========================================
        # 6. Decision Tree Classifier
        # ==========================================
        dt_score = 4.8
        dt_reasons = []
        dt_warnings = []
        dt_params = {"max_depth": 5, "random_state": 42}

        dt_reasons.append("Offers complete interpretability with hierarchical IF-THEN rule visualization.")
        dt_score += 1.0

        if has_mixed:
            dt_reasons.append("Naturally handles mixed continuous and categorical feature splits.")
            dt_score += 0.8

        if n_rows < 1000:
            dt_reasons.append("Fast training on small samples.")
            dt_score += 0.5

        if is_imbalanced:
            dt_params["class_weight"] = "balanced"
            dt_reasons.append("Supports class_weight='balanced' to adjust Gini impurity weighting.")
            dt_score += 0.5

        dt_warnings.append("Single decision trees exhibit high variance and are prone to overfitting without pruning.")
        dt_score -= 0.5

        dt_score = min(max(round(dt_score, 1), 3.0), 8.0)
        candidates.append({
            "algorithm": "Decision Tree Classifier",
            "model_key": "DecisionTreeClassifier",
            "score": dt_score,
            "reasons": dt_reasons,
            "warnings": dt_warnings,
            "preprocessing_requirements": "Median/Mode imputation + One-Hot Encoding (No feature scaling needed).",
            "recommended_params": dt_params,
            "requires_scaling": False
        })

        # ==========================================
        # 7. Naive Bayes (GaussianNB)
        # ==========================================
        nb_score = 4.5
        nb_reasons = []
        nb_warnings = []
        nb_params = {}

        if n_cat == 0 and n_num > 0:
            nb_reasons.append("Fast probabilistic generative model assuming Gaussian feature distributions.")
            nb_score += 1.0

        if n_feats >= 50:
            nb_reasons.append("Linear scaling per feature enables ultra-fast inference in high dimensions.")
            nb_score += 0.8

        if n_rows < 500:
            nb_reasons.append("Very low variance baseline on small data.")
            nb_score += 0.6

        nb_warnings.append("Strong conditional independence assumption rarely holds in correlated tabular datasets.")
        nb_score -= 0.8

        nb_score = min(max(round(nb_score, 1), 2.5), 7.8)
        candidates.append({
            "algorithm": "Naive Bayes (Gaussian)",
            "model_key": "GaussianNB",
            "score": nb_score,
            "reasons": nb_reasons or ["Fast, lightweight generative baseline."],
            "warnings": nb_warnings,
            "preprocessing_requirements": "Median imputation + One-Hot Encoding + StandardScaler.",
            "recommended_params": nb_params,
            "requires_scaling": True
        })

        return candidates

    def _score_regression_algorithms(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculates explainable scores for candidate regression algorithms."""
        n_rows = meta.get("number_of_rows", 100)
        n_feats = meta.get("number_of_features", 10)
        n_num = meta.get("numerical_features_count", 0)
        n_cat = meta.get("categorical_features_count", 0)
        has_mixed = meta.get("has_mixed_types", False)
        missing_pct = meta.get("missing_values_percentage", 0.0)
        p_to_n = meta.get("feature_to_sample_ratio", 0.01)

        candidates = []

        # 1. Gradient Boosting Regressor
        gbr_score = 7.0
        gbr_reasons = ["Minimizes continuous loss functions via gradient descent on residual errors."]
        gbr_warnings = []
        if n_rows >= 500:
            gbr_score += 1.2
            gbr_reasons.append("Dataset volume provides sufficient gradients for stable boosting convergence.")
        if has_mixed:
            gbr_score += 0.8
            gbr_reasons.append("Handles non-linear relationships across mixed feature types.")
        gbr_score = min(round(gbr_score, 1), 9.6)
        candidates.append({
            "algorithm": "Gradient Boosting Regressor",
            "model_key": "GradientBoostingRegressor",
            "score": gbr_score,
            "reasons": gbr_reasons,
            "warnings": gbr_warnings or ["Can be sensitive to large target outliers."],
            "preprocessing_requirements": "Imputation + One-Hot Encoding.",
            "recommended_params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
            "requires_scaling": False
        })

        # 2. Random Forest Regressor
        rfr_score = 6.8
        rfr_reasons = ["Averages predictions across multiple deep decision trees to suppress prediction variance."]
        rfr_warnings = []
        if has_mixed or n_cat > 0:
            rfr_score += 1.0
            rfr_reasons.append("Excels at mixed numerical and categorical continuous target mapping.")
        if n_rows < 5000:
            rfr_score += 0.8
            rfr_reasons.append("Robust against overfitting on moderate sample sizes.")
        rfr_score = min(round(rfr_score, 1), 9.4)
        candidates.append({
            "algorithm": "Random Forest Regressor",
            "model_key": "RandomForestRegressor",
            "score": rfr_score,
            "reasons": rfr_reasons,
            "warnings": rfr_warnings or ["Cannot extrapolate continuous targets beyond training min/max bounds."],
            "preprocessing_requirements": "Imputation + One-Hot Encoding.",
            "recommended_params": {"n_estimators": 100, "random_state": 42},
            "requires_scaling": False
        })

        # 3. Ridge Regression
        ridge_score = 5.5
        ridge_reasons = ["L2 regularized least-squares regression with analytical closed-form solution."]
        ridge_warnings = []
        if n_feats >= 20 or p_to_n > 0.05:
            ridge_score += 1.6
            ridge_reasons.append("L2 penalty prevents coefficient explosion in collinear or high-dimensional features.")
        if n_cat == 0:
            ridge_score += 0.8
            ridge_reasons.append("Purely numerical feature space provides smooth linear response surfaces.")
        ridge_score = min(round(ridge_score, 1), 8.8)
        candidates.append({
            "algorithm": "Ridge Regression",
            "model_key": "Ridge",
            "score": ridge_score,
            "reasons": ridge_reasons,
            "warnings": ridge_warnings or ["Assumes linear target response relationship."],
            "preprocessing_requirements": "Imputation + One-Hot Encoding + Mandatory StandardScaler.",
            "recommended_params": {"alpha": 1.0},
            "requires_scaling": True
        })

        # 4. Linear Regression
        lr_score = 5.0
        lr_reasons = ["Standard Ordinary Least Squares (OLS) baseline."]
        lr_warnings = []
        if n_rows < 1000 and n_feats < 15 and n_cat == 0:
            lr_score += 1.5
            lr_reasons.append("Low feature-to-sample ratio allows unbiased parameter estimation without severe multicollinearity.")
        if p_to_n > 0.1:
            lr_warnings.append("High feature-to-sample ratio risks singular covariance matrices; Ridge is preferred.")
            lr_score -= 1.5
        lr_score = min(max(round(lr_score, 1), 3.0), 8.2)
        candidates.append({
            "algorithm": "Linear Regression (OLS)",
            "model_key": "LinearRegression",
            "score": lr_score,
            "reasons": lr_reasons,
            "warnings": lr_warnings or ["Highly sensitive to multicollinearity and leverage outliers."],
            "preprocessing_requirements": "Imputation + One-Hot Encoding + Mandatory StandardScaler.",
            "recommended_params": {},
            "requires_scaling": True
        })

        # 5. Decision Tree Regressor
        dtr_score = 4.5
        dtr_reasons = ["Non-linear piecewise constant approximation with clear decision thresholds."]
        dtr_warnings = ["Prone to high step-function approximation variance without ensembling."]
        candidates.append({
            "algorithm": "Decision Tree Regressor",
            "model_key": "DecisionTreeRegressor",
            "score": dtr_score,
            "reasons": dtr_reasons,
            "warnings": dtr_warnings,
            "preprocessing_requirements": "Imputation + One-Hot Encoding.",
            "recommended_params": {"max_depth": 5, "random_state": 42},
            "requires_scaling": False
        })

        # 6. KNN Regressor
        knnr_score = 4.0
        knnr_reasons = ["Predicts target values by averaging the targets of the k-nearest Euclidean neighbors."]
        knnr_warnings = []
        if n_feats >= 20:
            knnr_warnings.append("Distance concentration in higher dimensions degrades neighbor locality.")
            knnr_score -= 1.0
        candidates.append({
            "algorithm": "K-Nearest Neighbors Regressor",
            "model_key": "KNeighborsRegressor",
            "score": knnr_score,
            "reasons": knnr_reasons,
            "warnings": knnr_warnings or ["Requires scaled features and uniform distance metrics."],
            "preprocessing_requirements": "Imputation + One-Hot Encoding + Mandatory StandardScaler.",
            "recommended_params": {"n_neighbors": 5},
            "requires_scaling": True
        })

        return candidates

    def _generate_holistic_explanation(
        self,
        meta: Dict[str, Any],
        top_rec: Dict[str, Any]
    ) -> str:
        """Generates dynamic natural language meta-routing synthesis."""
        n_rows = meta.get("number_of_rows", 0)
        n_feats = meta.get("number_of_features", 0)
        size_cat = meta.get("dataset_size", "Medium")
        dim_cat = meta.get("dimensionality", "Low")
        target_type = meta.get("target_type", "classification")
        has_mixed = meta.get("has_mixed_types", False)
        missing_pct = meta.get("missing_values_percentage", 0.0)
        top_algo = top_rec["algorithm"]
        top_score = top_rec["score"]

        target_meta = meta.get("target_details", {})
        sub_type = target_meta.get("sub_type", "")
        imbalance = target_meta.get("is_imbalanced", False)
        imbalance_ratio = target_meta.get("class_imbalance_ratio", 1.0)

        explanation = (
            f"The analyzed dataset contains **{n_rows:,} instances** across **{n_feats} features**, "
            f"categorized as a **{size_cat}** dataset with **{dim_cat}** dimensionality. "
        )

        if target_type == "classification":
            explanation += f"The target represents a **{sub_type} classification** task. "
            if imbalance:
                explanation += f"A significant class imbalance ratio of **{imbalance_ratio}:1** was detected. "
            else:
                explanation += "The target classes are reasonably balanced. "
        else:
            explanation += "The target variable is continuous, forming a **regression** problem. "

        if has_mixed:
            explanation += (
                f"Features are heterogeneous ({meta.get('numerical_features_count')} numeric, "
                f"{meta.get('categorical_features_count')} categorical). "
            )
        else:
            explanation += f"Features are homogeneously {'numerical' if meta.get('numerical_features_count') > 0 else 'categorical'}. "

        if missing_pct > 0:
            explanation += f"Data quality analysis identified **{missing_pct}% missing values**, requiring robust imputation. "
        else:
            explanation += "No missing values were detected. "

        explanation += (
            f"Based on these meta-features, the Meta-Learning Strategy Router awarded **{top_algo}** "
            f"the highest recommendation score of **{top_score}/10** because it optimally balances "
            f"representation capacity, non-linear feature interaction modeling, and generalization stability."
        )

        return explanation
