"""
Meta-Learning Machine Learning Strategy Router - Streamlit Academic Dashboard
An Explainable Dataset-Aware Machine Learning Algorithm Recommendation System
"""

import os
import json
import io
import time
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Internal package imports
from src.dataset_loader import (
    load_openml_dataset, load_csv_dataset, DATASET_REGISTRY, get_dataset_dir
)
from src.dataset_analyzer import analyze_dataset, suggest_target_column
from src.strategy_router import StrategyRouter
from src.model_trainer import train_and_cross_validate_candidates
from src.evaluator import evaluate_trained_models
from src.visualizations import (
    plot_dataset_characteristics, plot_missing_values, plot_feature_types,
    plot_recommendation_scores, plot_model_comparison, plot_confusion_matrix,
    plot_actual_vs_predicted, generate_all_visualizations, get_outputs_dir
)

# Configure Streamlit page
st.set_page_config(
    page_title="Meta-Learning Strategy Router",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic Theme CSS
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .metric-card-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }
    .badge-rank1 {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-secondary {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .explanation-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 15px;
        margin-bottom: 20px;
        color: #1E3A8A;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .step-node {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 8px 12px;
        text-align: center;
        font-weight: 600;
        font-size: 0.82rem;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes Streamlit session states for persistent interactive workflow."""
    if "active_dataset_key" not in st.session_state:
        st.session_state.active_dataset_key = None
    if "dataset_df" not in st.session_state:
        st.session_state.dataset_df = None
    if "dataset_metadata" not in st.session_state:
        st.session_state.dataset_metadata = {}
    if "selected_target" not in st.session_state:
        st.session_state.selected_target = None
    if "meta_features" not in st.session_state:
        st.session_state.meta_features = None
    if "routing_results" not in st.session_state:
        st.session_state.routing_results = None
    if "training_output" not in st.session_state:
        st.session_state.training_output = None
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None


init_session_state()

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=60)
    st.markdown("### ⚙️ Dataset & Pipeline Config")
    
    data_source_mode = st.radio(
        "Select Dataset Source:",
        ["OpenML Benchmark Dataset", "Custom OpenML ID", "Upload CSV File"],
        index=0
    )

    dataset_needs_load = False
    new_dataset_key = None

    if data_source_mode == "OpenML Benchmark Dataset":
        selected_benchmark = st.selectbox(
            "Benchmark Dataset:",
            list(DATASET_REGISTRY.keys()),
            index=0,  # Titanic is index 0
            help="Select one of the standard benchmark datasets."
        )
        max_samples = st.slider("Max Training Samples (MNIST/Large):", min_value=500, max_value=10000, value=5000, step=500)
        new_dataset_key = f"benchmark_{selected_benchmark}_{max_samples}"

        # Automatic instantaneous load when selection changes or first run
        if st.session_state.active_dataset_key != new_dataset_key:
            dataset_needs_load = True

    elif data_source_mode == "Custom OpenML ID":
        custom_id = st.number_input("OpenML Dataset ID:", min_value=1, max_value=100000, value=40945, step=1)
        max_samples = st.slider("Max Training Samples:", min_value=500, max_value=20000, value=5000, step=500)
        new_dataset_key = f"custom_openml_{custom_id}_{max_samples}"
        
        if st.session_state.active_dataset_key != new_dataset_key:
            if st.button("🔍 Fetch OpenML Dataset", use_container_width=True, type="primary"):
                dataset_needs_load = True

    else:
        uploaded_file = st.file_uploader("Upload your CSV dataset:", type=["csv"])
        if uploaded_file is not None:
            new_dataset_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.active_dataset_key != new_dataset_key:
                dataset_needs_load = True

    st.markdown("---")
    st.markdown("### 🎛️ Routing Parameters")
    top_k_candidates = st.slider("Top Recommended Candidates to Train:", min_value=2, max_value=5, value=3)
    cv_fold_count = st.slider("Cross-Validation Folds:", min_value=3, max_value=10, value=5)
    
    st.markdown("---")
    st.markdown("""
    **Project Info**  
    🎓 **Level:** B.Tech AI / ML  
    🔬 **Approach:** Dataset-Aware Meta-Feature Routing  
    📊 **Engine:** Scikit-Learn + Heuristic Multi-Criteria Scoring  
    """)

# ==============================================================================
# REACTIVE DATASET LOADER & STATE ISOLATION
# ==============================================================================
if dataset_needs_load and new_dataset_key is not None:
    with st.spinner("Loading dataset and initializing meta-features..."):
        if data_source_mode == "OpenML Benchmark Dataset":
            loaded_df, default_target, metadata = load_openml_dataset(
                selected_benchmark, max_samples=max_samples
            )
        elif data_source_mode == "Custom OpenML ID":
            loaded_df, default_target, metadata = load_openml_dataset(
                int(custom_id), max_samples=max_samples
            )
        else:
            loaded_df, default_target, metadata = load_csv_dataset(uploaded_file)

        # Set active dataset and reset ALL downstream training/evaluation to prevent contamination
        st.session_state.active_dataset_key = new_dataset_key
        st.session_state.dataset_df = loaded_df
        st.session_state.dataset_metadata = metadata
        st.session_state.selected_target = default_target
        
        # Immediate auto-analysis for seamless UX
        meta = analyze_dataset(loaded_df, target_column=default_target)
        router = StrategyRouter()
        routing = router.route_and_recommend(meta, top_k=top_k_candidates)

        st.session_state.meta_features = meta
        st.session_state.routing_results = routing
        st.session_state.training_output = None
        st.session_state.evaluation_results = None

# Fallback initial load if state was empty
if st.session_state.dataset_df is None:
    loaded_df, default_target, metadata = load_openml_dataset("Titanic", max_samples=5000)
    st.session_state.active_dataset_key = "benchmark_Titanic_5000"
    st.session_state.dataset_df = loaded_df
    st.session_state.dataset_metadata = metadata
    st.session_state.selected_target = default_target
    meta = analyze_dataset(loaded_df, target_column=default_target)
    router = StrategyRouter()
    routing = router.route_and_recommend(meta, top_k=top_k_candidates)
    st.session_state.meta_features = meta
    st.session_state.routing_results = routing

df = st.session_state.dataset_df

# ==============================================================================
# MAIN PAGE HEADER & WORKFLOW FLOW DIAGRAM
# ==============================================================================
st.markdown('<div class="main-header">META-LEARNING MACHINE LEARNING STRATEGY ROUTER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An Explainable Dataset-Aware Machine Learning Algorithm Recommendation System</div>', unsafe_allow_html=True)

# Visual Workflow Diagram
st.markdown("""
<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:12px 16px; margin-bottom:20px;">
    <div style="font-size:0.78rem; font-weight:700; color:#64748B; margin-bottom:8px; text-transform:uppercase;">
        System Architecture & Execution Pipeline
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:6px;">
        <span class="step-node">📁 1. Dataset Ingestion</span>
        <span style="color:#94A3B8; font-weight:bold;">→</span>
        <span class="step-node">🔍 2. Meta-Feature Extraction</span>
        <span style="color:#94A3B8; font-weight:bold;">→</span>
        <span class="step-node">🎯 3. Target Type Detection</span>
        <span style="color:#94A3B8; font-weight:bold;">→</span>
        <span class="step-node">🧠 4. Strategy Router Scoring</span>
        <span style="color:#94A3B8; font-weight:bold;">→</span>
        <span class="step-node">⚡ 5. Model Training (5-Fold CV)</span>
        <span style="color:#94A3B8; font-weight:bold;">→</span>
        <span class="step-node">🏆 6. Evaluation & Best Model</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# DATASET LOADING / TARGET SELECTION AREA
# ==============================================================================
st.markdown("### 📊 Active Dataset & Target Configuration")
col_preview1, col_preview2 = st.columns([3, 1])

with col_preview1:
    meta_info = st.session_state.dataset_metadata
    st.markdown(f"**Active Dataset:** `{meta_info.get('dataset_name', 'Dataset')}` | **Source:** `{meta_info.get('source', 'Unknown')}` | **Loaded Records:** `{len(df):,}` rows × `{len(df.columns)}` columns")
    with st.expander(f"👁️ View {meta_info.get('dataset_name', '')} Dataset Preview (First 10 Rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

with col_preview2:
    all_columns = list(df.columns)
    suggested = st.session_state.selected_target or suggest_target_column(df)
    target_idx = all_columns.index(suggested) if suggested in all_columns else len(all_columns) - 1
    
    selected_target = st.selectbox(
        "🎯 Select Target Column:",
        all_columns,
        index=target_idx,
        help="Select the column you wish the ML models to predict."
    )
    
    # If user changes target column, re-analyze automatically
    if selected_target != st.session_state.selected_target:
        st.session_state.selected_target = selected_target
        meta = analyze_dataset(df, target_column=selected_target)
        router = StrategyRouter()
        routing = router.route_and_recommend(meta, top_k=top_k_candidates)
        st.session_state.meta_features = meta
        st.session_state.routing_results = routing
        st.session_state.training_output = None
        st.session_state.evaluation_results = None

    if st.button("🚀 Re-Analyze Dataset & Route", type="primary", use_container_width=True):
        with st.spinner("Extracting meta-features and scoring algorithms..."):
            meta = analyze_dataset(df, target_column=selected_target)
            router = StrategyRouter()
            routing = router.route_and_recommend(meta, top_k=top_k_candidates)

            st.session_state.meta_features = meta
            st.session_state.routing_results = routing
            st.session_state.training_output = None
            st.session_state.evaluation_results = None
            st.success("Analysis and Strategy Routing updated!")

st.markdown("---")

# ==============================================================================
# DATASET CHARACTERISTICS METRIC CARDS
# ==============================================================================
if st.session_state.meta_features is not None:
    meta = st.session_state.meta_features
    routing = st.session_state.routing_results

    st.markdown("### 📋 Dataset Meta-Feature Summary")
    
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Instances (n)</div>
            <div class="metric-card-value">{meta['number_of_rows']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Features (p)</div>
            <div class="metric-card-value">{meta['number_of_features']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Feature Types</div>
            <div class="metric-card-value" style="font-size:1.15rem;">{meta['numerical_features_count']} Num / {meta['categorical_features_count']} Cat</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Missing Values</div>
            <div class="metric-card-value">{meta['missing_values_percentage']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        sub_t = meta['target_details'].get('sub_type', '')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Target Type</div>
            <div class="metric-card-value" style="font-size:1.15rem; color:#2563EB;">{meta['target_type'].upper()} ({sub_t})</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Scale / Dim</div>
            <div class="metric-card-value" style="font-size:1.15rem;">{meta['dataset_size']} / {meta['dimensionality']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # TABBED INTERFACE
    # ==============================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 1. Dataset Analysis & Visuals",
        "🧠 2. Strategy Router & Ranking",
        "🚀 3. Model Training & Metrics",
        "🎯 4. Diagnostic Plots & Confusion Matrix",
        "📋 5. Benchmark Meta-Comparison"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: DATASET ANALYSIS & META-FEATURES
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown(f"#### 🔍 Meta-Feature Profile: {st.session_state.dataset_metadata.get('dataset_name', '')}")
        st.markdown("""
        Meta-features represent high-level structural, statistical, and information-theoretic properties of the dataset.
        These features form the input vector for algorithm selection rules.
        """)

        t1_col1, t1_col2 = st.columns(2)
        with t1_col1:
            st.markdown("##### 📈 Feature Modality Breakdown")
            fig_ft = plot_feature_types(meta)
            st.pyplot(fig_ft)

        with t1_col2:
            st.markdown("##### ⚠️ Missing Value Distribution")
            fig_mv = plot_missing_values(meta)
            if fig_mv:
                st.pyplot(fig_mv)
            else:
                st.info("No missing values found.")

        st.markdown("##### 📌 Detailed Meta-Feature Attributes Table")
        meta_table_data = {
            "Meta-Feature": [
                "Instance Count (n)", "Feature Count (p)", "Feature-to-Sample Ratio (p/n)",
                "Numerical Features", "Categorical Features", "Total Missing Cells",
                "Missing Values (%)", "Duplicate Rows", "Target Column Name",
                "Target Paradigm", "Target Specifics", "Class Imbalance Ratio", "Scale Category"
            ],
            "Extracted Value": [
                f"{meta['number_of_rows']:,}",
                str(meta['number_of_features']),
                str(meta['feature_to_sample_ratio']),
                f"{meta['numerical_features_count']} features ({', '.join(meta['numerical_features'][:4]) + ('...' if len(meta['numerical_features']) > 4 else '')})",
                f"{meta['categorical_features_count']} features ({', '.join(meta['categorical_features'][:4]) + ('...' if len(meta['categorical_features']) > 4 else '')})",
                str(meta['missing_values_count']),
                f"{meta['missing_values_percentage']}%",
                f"{meta['duplicate_rows']} ({meta['duplicate_rows_percentage']}%)",
                str(meta['target_column']),
                meta['target_type'].capitalize(),
                f"{meta['target_details'].get('sub_type', 'N/A')} ({meta['target_details'].get('number_of_classes', 'N/A')} classes: {', '.join([str(c) for c in meta['target_details'].get('classes', [])[:5]])})" if meta['target_type'] == "classification" else "Continuous numeric target",
                f"{meta['target_details'].get('class_imbalance_ratio', 1.0)}:1" if meta['target_type'] == "classification" else "N/A",
                f"{meta['dataset_size']} scale, {meta['dimensionality']} dimensionality"
            ]
        }
        st.dataframe(pd.DataFrame(meta_table_data), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: STRATEGY ROUTER RECOMMENDATIONS
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("#### 🧠 Meta-Learning Strategy Router Scoring Engine")
        st.markdown("""
        The Strategy Router applies explainable multi-criteria heuristic scoring to evaluate how well each candidate
        machine learning model fits the extracted dataset meta-features.
        """)

        # Holistic Explanation Banner
        st.markdown(f"""
        <div class="explanation-box">
            <strong>🤖 Strategy Router Synthesis:</strong><br>
            {routing['holistic_explanation']}
        </div>
        """, unsafe_allow_html=True)

        t2_col1, t2_col2 = st.columns([1, 1.2])

        with t2_col1:
            st.markdown("##### 📊 Recommendation Scores Chart")
            fig_recs = plot_recommendation_scores(routing["all_ranked_algorithms"])
            st.pyplot(fig_recs)

        with t2_col2:
            st.markdown("##### 🏆 Ranked Candidate Algorithm Breakdown")
            for rec in routing["all_ranked_algorithms"]:
                is_top = (rec["rank"] == 1)
                badge_html = '<span class="badge-rank1">⭐ Rank 1 — Top Recommendation</span>' if is_top else f'<span class="badge-secondary">Rank {rec["rank"]}</span>'
                
                with st.expander(f"Rank {rec['rank']}: {rec['algorithm']} — Score: {rec['score']}/10.0", expanded=is_top):
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.markdown(f"**Fitness Score:** `{rec['score']} / 10.0`")
                    st.markdown("**Why Recommended:**")
                    for reason in rec["reasons"]:
                        st.markdown(f"- ✅ {reason}")
                    if rec["warnings"]:
                        st.markdown("**Caveats & Tuning Considerations:**")
                        for warning in rec["warnings"]:
                            st.markdown(f"- ⚠️ {warning}")
                    st.markdown(f"**Preprocessing Requirements:** `{rec['preprocessing_requirements']}`")
                    st.markdown(f"**Recommended Hyperparameters:** `{rec.get('recommended_params', {})}`")

    # --------------------------------------------------------------------------
    # TAB 3: MODEL TRAINING & EVALUATION
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("#### 🚀 Candidate Model Training & Cross-Validation")
        st.markdown(f"""
        Train the top **{len(routing['top_recommendations'])} recommended candidate pipelines** on dataset **{st.session_state.dataset_metadata.get('dataset_name', '')}** using **{cv_fold_count}-Fold Cross-Validation**
        and holdout test evaluation (80/20 train/test split).
        """)

        train_col1, train_col2 = st.columns([2, 1])
        with train_col1:
            train_btn = st.button(f"⚡ Train Top Recommended Models for {st.session_state.dataset_metadata.get('dataset_name', '')}", type="primary", use_container_width=True)

        if train_btn or st.session_state.training_output is not None:
            if train_btn:
                with st.spinner(f"Fitting pipelines on {st.session_state.dataset_metadata.get('dataset_name', '')} and calculating {cv_fold_count}-Fold cross-validation scores..."):
                    train_out = train_and_cross_validate_candidates(
                        df=df,
                        target_column=selected_target,
                        meta_features=meta,
                        top_candidates=routing["top_recommendations"],
                        test_size=0.2,
                        cv_folds=cv_fold_count
                    )
                    eval_out = evaluate_trained_models(train_out)
                    st.session_state.training_output = train_out
                    st.session_state.evaluation_results = eval_out
                    
                    # Auto-generate and save all outputs
                    generate_all_visualizations(meta, routing, eval_out)
                    st.success(f"Training and evaluation for {st.session_state.dataset_metadata.get('dataset_name', '')} completed successfully!")

            eval_res = st.session_state.evaluation_results

            if eval_res is not None:
                st.markdown("---")
                st.markdown("##### 📈 Model Performance Comparison Table")

                records = []
                if eval_res["target_type"] == "classification":
                    for m in eval_res["evaluated_models"]:
                        if m.get("status") == "SUCCESS":
                            met = m["metrics"]
                            records.append({
                                "Router Rank": f"Rank {m['router_rank']}",
                                "Algorithm": m["algorithm"],
                                "Router Score": f"{m['router_score']}/10",
                                f"{cv_fold_count}-Fold CV Accuracy": f"{met['cv_mean']:.4f} ± {met['cv_std']:.4f}",
                                "Test Accuracy": f"{met['accuracy']:.4f}",
                                "Weighted Precision": f"{met['precision']:.4f}",
                                "Weighted Recall": f"{met['recall']:.4f}",
                                "Weighted F1-Score": f"{met['f1_score']:.4f}",
                                "Training Time (s)": f"{met['training_time']:.4f}s"
                            })
                else:
                    for m in eval_res["evaluated_models"]:
                        if m.get("status") == "SUCCESS":
                            met = m["metrics"]
                            records.append({
                                "Router Rank": f"Rank {m['router_rank']}",
                                "Algorithm": m["algorithm"],
                                "Router Score": f"{m['router_score']}/10",
                                f"{cv_fold_count}-Fold CV R²": f"{met['cv_mean']:.4f} ± {met['cv_std']:.4f}",
                                "Test R² Score": f"{met['r2_score']:.4f}",
                                "MAE": f"{met['mae']:.4f}",
                                "RMSE": f"{met['rmse']:.4f}",
                                "Training Time (s)": f"{met['training_time']:.4f}s"
                            })

                st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

                best_m = eval_res.get("best_model")
                if best_m:
                    st.markdown(f"""
                    <div style="background:#ECFDF5; border:1px solid #10B981; border-radius:8px; padding:14px 18px; margin-top:15px;">
                        <h4 style="color:#065F46; margin:0 0 6px 0;">🏆 Best Performing Model: {best_m['algorithm']}</h4>
                        <p style="color:#047857; margin:0;">
                            Achieved highest validation fitness with <strong>{best_m['metrics'].get('accuracy', best_m['metrics'].get('r2_score')):.4f}</strong> performance score 
                            in <strong>{best_m['metrics']['training_time']:.4f}s</strong> training time.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("##### 📊 Comparative Performance Chart")
                fig_comp = plot_model_comparison(eval_res)
                st.pyplot(fig_comp)

    # --------------------------------------------------------------------------
    # TAB 4: CONFUSION MATRIX & DIAGNOSTIC PLOTS
    # --------------------------------------------------------------------------
    with tab4:
        eval_res = st.session_state.evaluation_results
        if eval_res is None or eval_res.get("best_model") is None:
            st.info(f"👉 Please train the candidate models in Tab 3 to inspect diagnostic plots and confusion matrix for {st.session_state.dataset_metadata.get('dataset_name', 'the dataset')}.")
        else:
            best_m = eval_res.get("best_model")
            t_type = eval_res.get("target_type")

            if t_type == "classification":
                st.markdown(f"#### 🎯 Confusion Matrix — Best Performing Model ({best_m['algorithm']})")
                st.markdown(f"Dataset: **{st.session_state.dataset_metadata.get('dataset_name', '')}** | Target: **`{st.session_state.selected_target}`** | Evaluated on unseen test set ({len(best_m['y_test'])} samples).")

                c_diag1, c_diag2 = st.columns([1.2, 1])
                with c_diag1:
                    fig_cm = plot_confusion_matrix(best_m)
                    st.pyplot(fig_cm)

                with c_diag2:
                    st.markdown("##### 📊 Best Model Performance Metrics")
                    b_met = best_m["metrics"]
                    st.metric("Test Accuracy", f"{b_met['accuracy']:.4f}")
                    st.metric("Weighted F1-Score", f"{b_met['f1_score']:.4f}")
                    st.metric("Weighted Precision", f"{b_met['precision']:.4f}")
                    st.metric("Weighted Recall", f"{b_met['recall']:.4f}")
                    st.metric("5-Fold CV Mean", f"{b_met['cv_mean']:.4f} (± {b_met['cv_std']:.4f})")
                    st.markdown(f"**Target Classes Analyzed:** `{', '.join(b_met.get('class_labels', []))}`")

            else:
                st.markdown(f"#### 📈 Actual vs Predicted Plot — Best Regressor ({best_m['algorithm']})")
                st.markdown(f"Dataset: **{st.session_state.dataset_metadata.get('dataset_name', '')}** | Target: **`{st.session_state.selected_target}`** | Evaluated on unseen test predictions.")

                c_diag1, c_diag2 = st.columns([1.2, 1])
                with c_diag1:
                    fig_reg = plot_actual_vs_predicted(best_m)
                    st.pyplot(fig_reg)

                with c_diag2:
                    st.markdown("##### 📊 Best Regressor Metrics")
                    b_met = best_m["metrics"]
                    st.metric("Test R² Score", f"{b_met['r2_score']:.4f}")
                    st.metric("Root Mean Squared Error (RMSE)", f"{b_met['rmse']:.4f}")
                    st.metric("Mean Absolute Error (MAE)", f"{b_met['mae']:.4f}")
                    st.metric("5-Fold CV Mean R²", f"{b_met['cv_mean']:.4f} (± {b_met['cv_std']:.4f})")

    # --------------------------------------------------------------------------
    # TAB 5: BENCHMARK META-LEARNING COMPARISON
    # --------------------------------------------------------------------------
    with tab5:
        st.markdown("#### 📋 OpenML Dataset Meta-Feature Comparison")
        st.markdown("""
        Demonstrating the core meta-learning paradigm: different dataset meta-feature vectors trigger distinct,
        tailored algorithm rankings.
        """)

        benchmark_comparison_data = [
            {
                "Dataset": "Titanic (40945)",
                "Rows (n)": "891",
                "Features (p)": "7",
                "Numerical": "5",
                "Categorical": "2",
                "Missing %": "20.1%",
                "Target Type": "Binary Classification",
                "Classes": "2 (0: Did not survive, 1: Survived)",
                "Scale": "Small",
                "Dimensionality": "Low",
                "Top Recommendation": "Random Forest Classifier (Score: 9.3/10)"
            },
            {
                "Dataset": "Iris (61)",
                "Rows (n)": "150",
                "Features (p)": "4",
                "Numerical": "4",
                "Categorical": "0",
                "Missing %": "0.0%",
                "Target Type": "Multiclass Classification",
                "Classes": "3 (setosa, versicolor, virginica)",
                "Scale": "Small",
                "Dimensionality": "Low",
                "Top Recommendation": "Support Vector Machine / Logistic Regression"
            },
            {
                "Dataset": "Breast Cancer (13)",
                "Rows (n)": "569",
                "Features (p)": "30",
                "Numerical": "30",
                "Categorical": "0",
                "Missing %": "0.0%",
                "Target Type": "Binary Classification",
                "Classes": "2 (malignant, benign)",
                "Scale": "Small",
                "Dimensionality": "Medium",
                "Top Recommendation": "Support Vector Machine / Random Forest"
            },
            {
                "Dataset": "Diabetes (37)",
                "Rows (n)": "768",
                "Features (p)": "8",
                "Numerical": "8",
                "Categorical": "0",
                "Missing %": "0.0%",
                "Target Type": "Binary Classification",
                "Classes": "2 (0: Negative, 1: Positive)",
                "Scale": "Small",
                "Dimensionality": "Low",
                "Top Recommendation": "Gradient Boosting / Random Forest"
            },
            {
                "Dataset": "MNIST Digits (554)",
                "Rows (n)": "70,000",
                "Features (p)": "784",
                "Numerical": "784",
                "Categorical": "0",
                "Missing %": "0.0%",
                "Target Type": "Multiclass Classification",
                "Classes": "10 (Digits 0-9)",
                "Scale": "Large",
                "Dimensionality": "High",
                "Top Recommendation": "Linear SVM / Logistic Regression"
            }
        ]
        st.dataframe(pd.DataFrame(benchmark_comparison_data), use_container_width=True, hide_index=True)

# ==============================================================================
# FOOTER & ARTIFACT EXPORT
# ==============================================================================
st.markdown("---")
f_col1, f_col2, f_col3 = st.columns([2, 1, 1])

with f_col1:
    st.markdown("💾 **Export Generated Academic Artifacts & Report**")
    st.caption("All publication figures and JSON results are persisted into `outputs/`.")

with f_col2:
    json_path = os.path.join(get_outputs_dir(), "router_results.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as jf:
            st.download_button(
                "📥 Download Results JSON",
                data=jf.read(),
                file_name="router_results.json",
                mime="application/json",
                use_container_width=True
            )

with f_col3:
    st.markdown("*(B.Tech AI/ML Project Demonstration)*")
