# META-LEARNING MACHINE LEARNING STRATEGY ROUTER
### *An Explainable Dataset-Aware Machine Learning Algorithm Recommendation System*

---

## 1. Project Overview & Abstract

In applied data science and artificial intelligence, practitioners and novices frequently encounter the **Model Selection Dilemma**: choosing which machine learning algorithm is best suited for an arbitrary dataset. Standard practices often rely on arbitrary trial-and-error, brute-force grid searching over hundreds of models, or defaulting to a single favorite algorithm regardless of the data's structural characteristics.

The **Meta-Learning Machine Learning Strategy Router** addresses this fundamental challenge by designing an **explainable, dataset-aware heuristic routing system**. Rather than blindly fitting a fixed algorithm to every dataset, this system extracts intrinsic dataset characteristics—termed **meta-features** (e.g., sample count $n$, feature dimension $p$, feature-to-sample ratio $p/n$, feature types, missing value ratios, and class imbalance)—and computes quantitative multi-criteria fitness scores for candidate machine learning algorithms. The router outputs human-interpretable justifications for each recommendation, executes leak-free 5-fold cross-validation on the top candidate models, and provides interactive diagnostic evaluations, including confusion matrices and performance dashboards.

```
+-----------------------------------------------------------------------------------+
|               META-LEARNING MACHINE LEARNING STRATEGY ROUTER                      |
|                                                                                   |
|  [ Dataset Ingestion ] ---> [ Meta-Feature Extraction ] ---> [ Target Analysis ]  |
|                                                                     |             |
|                                                                     v             |
|  [ Model Training (5-Fold CV) ] <--- [ Top Candidates ] <--- [ Strategy Router ]  |
|               |                                                                   |
|               v                                                                   |
|  [ Evaluation Metrics ] ---> [ Confusion Matrix Heatmap ] ---> [ Visual Reports ] |
+-----------------------------------------------------------------------------------+
```

---

## 2. Problem Statement

Machine learning algorithms exhibit distinct mathematical assumptions, inductive biases, and computational complexities:
- **Support Vector Machines (SVM)** excel at finding maximum-margin linear boundaries in high-dimensional continuous spaces ($p \gg n$) but suffer from $O(n^2)$ to $O(n^3)$ training complexity on large datasets.
- **K-Nearest Neighbors (KNN)** is intuitive for small, dense numerical clusters, but degrades exponentially in high dimensions due to the *curse of dimensionality* ($d(\mathbf{x}_i, \mathbf{x}_j)$ becomes uniform) and incurs heavy inference latency $O(n \cdot d)$.
- **Tree-Based Ensembles (Random Forest, Gradient Boosting)** handle non-linear interactions and heterogeneous feature spaces (mixed numeric and categorical) exceptionally well, yet can be computationally unnecessary for simple linearly separable problems.
- **Linear Models (Logistic / Ridge Regression)** provide strong, well-regularized baselines when features are purely continuous or high-dimensional, but lack expressivity for complex multi-modal manifolds without explicit basis expansion.

Beginners lack the systematic meta-knowledge required to match these algorithm characteristics to their dataset. This project designs and implements an automated, explainable meta-learning strategy router to guide algorithm selection with mathematical and structural justification.

---

## 3. Key Objectives

1. **Automated Ingestion & Fallback:** Seamlessly fetch datasets from OpenML by ID or parse user-uploaded CSV files, backed by an offline fallback mechanism.
2. **Comprehensive Meta-Feature Extraction:** Analyze dataset dimensionality, feature modalities (continuous vs. discrete), missing value rates, and class distribution skews.
3. **Automated Target Detection:** Accurately discern between Classification (Binary / Multiclass) and Regression tasks.
4. **Explainable Multi-Criteria Scoring:** Compute normalized fitness scores ($0.0 - 10.0$) with transparent, rule-based justifications and tuning warnings.
5. **Leak-Free Preprocessing Pipelines:** Construct `sklearn.pipeline.Pipeline` with `ColumnTransformer` (median/mode imputation, one-hot encoding, and conditional `StandardScaler` only when required by distance-based/linear estimators).
6. **Rigorous Empirical Evaluation:** Train top-$K$ recommended models using 5-Fold Stratified K-Fold Cross-Validation, reporting CV means, test accuracy, precision, recall, F1-score, MAE, RMSE, and $R^2$.
7. **Authentic Diagnostic Visualizations:** Generate publication-ready figures (infographic cards, missing value bars, donut charts, score rankings, and actual test confusion matrices) saved to `outputs/`.
8. **Interactive Academic Dashboard:** Provide a Streamlit UI for hands-on experimentation, benchmark meta-feature comparisons, and artifact downloads.

---

## 4. System Architecture

The following diagram illustrates the component flow of the Meta-Learning Strategy Router:

```
                      +-----------------------------+
                      |   OpenML API / Local CSV    |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |     src/dataset_loader.py   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    src/dataset_analyzer.py  |
                      |   (Meta-Feature Extractor)  |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    src/strategy_router.py   |
                      |   (Scoring & Ranking Logic) |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |     src/preprocessing.py    |
                      | (ColumnTransformer/Pipeline)|
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |     src/model_trainer.py    |
                      |  (5-Fold CV + Train Split)  |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |       src/evaluator.py      |
                      | (Metrics & Confusion Matrix)|
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    src/visualizations.py    |
                      | (7 PNG Artifacts + JSON Rep)|
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Streamlit App / CLI UI   |
                      |   (app.py / main.py)        |
                      +-----------------------------+
```

---

## 5. Benchmark Datasets Analyzed

The project incorporates five diverse benchmark datasets from OpenML, representing distinct meta-feature vectors:

| Dataset | OpenML ID | Rows ($n$) | Features ($p$) | Modality | Missing % | Target Task | Distinctive Meta-Feature |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Titanic** | `40945` | 891 | 7 | Mixed (Num + Cat) | ~20.1% | Binary Classification | Heterogeneous features with missing ages |
| **Iris** | `61` | 150 | 4 | Pure Numeric | 0.0% | Multiclass (3 classes) | Small sample, purely continuous, low $p$ |
| **Breast Cancer** | `13` | 569 | 30 | Pure Numeric | 0.0% | Binary Classification | Moderate dimensionality, continuous geometric data |
| **Diabetes** | `37` | 768 | 8 | Pure Numeric | 0.0% | Binary Classification | Physiological numeric metrics, moderate non-linearity |
| **MNIST Digits** | `554` | 70,000 | 784 | High-Dim Pixel | 0.0% | Multiclass (10 classes) | Extremely high dimensionality ($p=784$), large $n$ |

---

## 6. Meta-Feature Taxonomy & Mathematical Formulation

The `src/dataset_analyzer.py` module extracts meta-features across five core dimensions:

### 1. Simple Dimensions
- **Sample Size ($n$):** Categorized as:
  - $\text{Small}: n < 1,000$
  - $\text{Medium}: 1,000 \le n < 10,000$
  - $\text{Large}: n \ge 10,000$
- **Feature Count ($p$):** Categorized as:
  - $\text{Low}: p < 20$
  - $\text{Medium}: 20 \le p < 100$
  - $\text{High}: p \ge 100$
- **Feature-to-Sample Ratio:**
  $$\gamma = \frac{p}{n}$$
  When $\gamma > 0.1$, high-dimensional regularization rules are activated.

### 2. Feature Type & Quality Meta-Features
- **Numerical Feature Count ($p_{\text{num}}$)** and **Categorical Feature Count ($p_{\text{cat}}$)**
- **Mixed Modality Flag:** $\text{IsMixed} = \mathbb{I}(p_{\text{num}} > 0 \land p_{\text{cat}} > 0)$
- **Missing Value Percentage:**
  $$\text{Missing\%} = \frac{\sum_{i=1}^n \sum_{j=1}^p \mathbb{I}(x_{ij} = \text{NaN})}{n \cdot p} \times 100$$

### 3. Target Distribution Meta-Features
- **Class Count ($K$):** Discerning Binary ($K=2$) vs. Multiclass ($K > 2$).
- **Imbalance Ratio ($IR$):**
  $$IR = \frac{\max_k |C_k|}{\min_k |C_k|}$$
  If $IR \ge 2.0$, models supporting `class_weight='balanced'` receive priority.

---

## 7. Strategy Router Scoring Logic

The Strategy Router (`src/strategy_router.py`) computes a fitness score $S(A, \mathcal{M}) \in [0.0, 10.0]$ for each candidate algorithm $A$ given meta-feature vector $\mathcal{M}$:

$$S(A, \mathcal{M}) = S_{\text{base}}(A) + \sum_{k} \Delta s_k(\text{rule}_k, \mathcal{M})$$

### Classification Algorithm Routing Rules

1. **Random Forest Classifier:**
   - Base Score: $6.0$
   - Non-linear modeling capability: $+1.2$
   - Mixed features / categorical presence: $+1.2$
   - Medium/Large sample size ($n \ge 1000$): $+0.8$
   - Class imbalance present ($IR \ge 2.0$): $+0.8$ (recommends `class_weight='balanced'`)
   - High dimensionality ($p \ge 100$): $+0.6$ (random subspace selection mitigates noise)

2. **Gradient Boosting Classifier:**
   - Base Score: $6.0$
   - Tabular boosting performance: $+1.5$
   - Medium sample size ($500 \le n < 20000$): $+1.0$
   - Very small sample ($n < 500$): $-0.6$ (overfitting warning)
   - Mixed feature interaction: $+0.8$

3. **Support Vector Machine (SVM):**
   - Base Score: $5.0$
   - High dimensionality ($p \ge 50$ or $\gamma > 0.05$): $+1.8$ (selects Linear kernel)
   - Small sample ($n < 1000$): $+1.4$ (selects RBF kernel)
   - Large sample ($n \ge 10000$): $-2.2$ ($O(n^2)$ training bottleneck warning)
   - Pure numerical features: $+0.8$

4. **Logistic Regression:**
   - Base Score: $5.0$
   - Binary classification: $+1.2$
   - High dimensionality ($p \ge 100$): $+1.5$ (L2 regularization generalizes well)
   - Small sample with pure numericals: $+0.8$
   - Mixed non-linear interactions: $-0.5$ (linear limit caveat)

5. **K-Nearest Neighbors (KNN):**
   - Base Score: $4.5$
   - Small, low-dimensional numerical data ($n < 1000, p < 20$): $+2.0$
   - High dimensionality ($p \ge 50$): $-2.0$ (distance concentration / curse of dimensionality)
   - Large dataset ($n \ge 10000$): $-1.8$ (slow inference query latency $O(n)$)
   - Categorical features: $-1.0$ (Euclidean distance distortion on one-hot flags)

---

## 8. Supported Algorithms & Hyperparameters

### Classification
1. **Random Forest Classifier** (`RandomForestClassifier(n_estimators=100, class_weight=...)`)
2. **Gradient Boosting Classifier** (`GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)`)
3. **Logistic Regression** (`LogisticRegression(max_iter=1000, penalty='l2')`)
4. **Support Vector Classifier** (`SVC(kernel='rbf'/'linear', probability=True)`)
5. **K-Nearest Neighbors** (`KNeighborsClassifier(n_neighbors=5)`)
6. **Decision Tree Classifier** (`DecisionTreeClassifier(max_depth=5)`)
7. **Gaussian Naive Bayes** (`GaussianNB()`)

### Regression
1. **Gradient Boosting Regressor** (`GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)`)
2. **Random Forest Regressor** (`RandomForestRegressor(n_estimators=100)`)
3. **Ridge Regression** (`Ridge(alpha=1.0)`)
4. **Linear Regression** (`LinearRegression()`)
5. **Decision Tree Regressor** (`DecisionTreeRegressor(max_depth=5)`)
6. **KNN Regressor** (`KNeighborsRegressor(n_neighbors=5)`)

---

## 9. Project Structure

```
Meta-Learning-Strategy-Router/
│
├── app.py                     # Streamlit Interactive Academic Dashboard
├── main.py                    # Command-Line Pipeline Runner & Verifier
├── generate_datasets.py       # Offline Dataset Generator / Pre-cacher
├── requirements.txt           # Project Dependencies
├── README.md                  # Comprehensive Documentation
│
├── datasets/                  # Local Benchmark CSV Files
│   ├── titanic.csv
│   ├── iris.csv
│   ├── breast_cancer.csv
│   ├── diabetes.csv
│   └── mnist.csv
│
├── src/                       # Modular Python Architecture
│   ├── __init__.py
│   ├── dataset_loader.py      # OpenML API + Offline Fallback Loader
│   ├── dataset_analyzer.py    # Meta-Feature Extraction Engine
│   ├── strategy_router.py     # Heuristic Recommendation Engine
│   ├── preprocessing.py       # Leakage-Free Scikit-Learn Pipelines
│   ├── model_trainer.py       # 5-Fold Stratified CV Trainer
│   ├── evaluator.py           # Metrics & Confusion Matrix Generator
│   └── visualizations.py      # Matplotlib/Seaborn Publication Plots
│
├── outputs/                   # Auto-generated Artifacts & Reports
│   ├── dataset_characteristics.png
│   ├── missing_values.png
│   ├── feature_types.png
│   ├── recommendation_chart.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── actual_vs_predicted.png
│   └── router_results.json
│
└── models/                    # Model Storage Directory
```

---

## 10. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- Internet connection (optional; system includes full offline fallbacks)

### Step 1: Clone or Navigate to Project
```bash
cd Meta-Learning-Strategy-Router
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 4: Run via Command Line Interface (CLI)
You can also run the complete pipeline and generate all outputs directly from the terminal:
```bash
# Run on Iris
python main.py --dataset Iris

# Run on Titanic
python main.py --dataset Titanic

# Run on Breast Cancer
python main.py --dataset "Breast Cancer"

# Run on Diabetes
python main.py --dataset Diabetes

# Run on MNIST (with sample limit for speed)
python main.py --dataset MNIST --max-samples 2000

# Run on any custom CSV file
python main.py --dataset path/to/your_dataset.csv --target your_target_column
```

---

## 11. Generated Output Artifacts

Running the pipeline automatically saves high-resolution publication-quality PNG charts and JSON reports into `outputs/`:

1. `dataset_characteristics.png` — Infographic summary of structural meta-features.
2. `missing_values.png` — Feature-wise missing value distribution bar chart.
3. `feature_types.png` — Donut chart of continuous vs. categorical feature split.
4. `recommendation_chart.png` — Horizontal ranking of candidate algorithm scores.
5. `model_comparison.png` — Grouped bar chart comparing CV Means, Test Accuracies, and F1-Scores.
6. `confusion_matrix.png` — Annotated heatmap of the best model's actual test predictions.
7. `actual_vs_predicted.png` — Scatter plot with identity fit line for regression tasks.
8. `router_results.json` — Machine-readable summary of meta-features, rankings, and test metrics.

---

## 12. Academic Integrity & Scientific Realism

> [!IMPORTANT]
> - **No Hard-Coded Results:** All evaluation scores, cross-validation metrics, training latencies, and confusion matrix cell values are computed dynamically by executing Scikit-Learn pipelines on actual train/test splits.
> - **No Data Leakage:** Preprocessors (imputers, scalers, and encoders) are fitted strictly on `X_train` within `sklearn.pipeline.Pipeline` objects and evaluated on unseen `X_test`.
> - **Accurate Terminology:** The router is clearly presented as an *explainable, heuristic dataset-aware meta-learning engine* rather than a black-box deep neural meta-learner.

---

## 13. Limitations & Future Scope

### Current Limitations
1. Heuristic rules are based on established statistical learning theory and empirical best practices; they are not learned from historical meta-loss gradients.
2. Hyperparameter optimization is performed using recommended defaults rather than extensive Bayesian optimization per candidate.

### Future Scope
1. **Trained Meta-Regressor:** Construct a meta-dataset by extracting meta-features across 1,000+ OpenML datasets and training an XGBoost/Neural meta-model to predict test accuracy directly.
2. **Automated Hyperparameter Scheduling:** Incorporate Optuna to perform budget-constrained hyperparameter tuning on the top recommended candidate.
3. **Multi-Modal Extension:** Expand meta-feature extraction to text (NLP embeddings) and image (CNN feature extractors) datasets.

---

## 14. Academic Viva & Defense Questions (Q&A)

**Q1: What distinguishes this project from a standard AutoML library (e.g., Auto-sklearn)?**  
*Answer:* Standard AutoML tools typically execute exhaustive black-box searches (evaluating dozens of models with genetic algorithms or Bayesian optimization). This project focuses on **explainable, instant meta-learning**: analyzing dataset meta-features *prior* to training and providing human-interpretable reasoning on *why* specific algorithms are structurally suited to the data.

**Q2: Why is feature scaling applied to Logistic Regression and SVM, but omitted for Random Forest?**  
*Answer:* Decision trees and tree ensembles partition feature spaces using orthogonal axis-aligned thresholds ($x_j \le \theta$). Monotonic scale transformations do not alter split entropy or Gini impurity. Conversely, distance-based and gradient-based algorithms (SVM, KNN, Logistic Regression) rely on geometric distances ($\|\mathbf{x}_i - \mathbf{x}_j\|_2$) and regularized gradient descent ($\frac{1}{2}\|\mathbf{w}\|_2^2$), making unscaled features dominate parameter updates.

**Q3: How does the router handle the Curse of Dimensionality?**  
*Answer:* When the feature-to-sample ratio $p/n > 0.1$ or $p \ge 50$, distance concentration causes all pairwise Euclidean distances to converge ($(\max d - \min d)/\min d \to 0$), severely degrading KNN. The router penalizes KNN by $-2.0$ points while promoting linear models with L2 regularization and Linear SVMs, which excel in high-dimensional hyperplanes.
