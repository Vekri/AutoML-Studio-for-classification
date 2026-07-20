"""AutoML Studio for Binary Classification — main application."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from config import FEATURE_SELECTION_METHODS, PROBLEM_DOMAINS
from modules.binning import apply_binning_to_dataframe, woe_iv_table
from modules.cleaning import apply_cleaning, auto_clean, generate_cleaning_recommendations
from modules.data_loader import (
    apply_column_selection,
    detect_binary_target,
    get_column_summary,
    load_csv,
    suggest_target_column,
)
from modules.export import (
    build_studio_manifest,
    dataframe_to_csv_bytes,
    dict_to_json_bytes,
    export_zip_bundle,
)
from modules.feature_selection import (
    get_selected_features,
    random_forest_importance,
    run_feature_selection,
)
from modules.validation import run_model_validation, validate_dataset
from modules.visualization import (
    box_plot_by_target,
    categorical_bar_chart,
    correlation_heatmap,
    feature_target_correlation_chart,
    get_categorical_columns,
    get_numeric_columns,
    missing_values_chart,
    numeric_histogram,
    target_distribution_chart,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoML Studio | Binary Classification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0; }
    .sub-header  { font-size: 1rem; color: #6B7280; margin-top: 0; }
    .metric-card { background: #F8FAFC; border-radius: 8px; padding: 1rem; border: 1px solid #E2E8F0; }
    .severity-high   { color: #DC2626; font-weight: 600; }
    .severity-medium { color: #D97706; font-weight: 600; }
    .severity-low    { color: #059669; font-weight: 600; }
    div[data-testid="stSidebar"] { background: #1E3A5F; }
    div[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "raw_df": None,
    "processed_df": None,
    "domain": "General Binary Classification",
    "target": None,
    "keep_cols": [],
    "drop_cols": [],
    "cleaning_config": {},
    "binning_config": {},
    "selected_features": [],
    "feature_selection_result": None,
    "validation_report": {},
    "model_validation": {},
    "recommendations": [],
    "step": 1,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 AutoML Studio")
    st.markdown("*Binary Classification Pipeline*")
    st.divider()

    steps = [
        "1 · Upload & Domain",
        "2 · Target & Columns",
        "3 · Data Cleaning",
        "4 · Visualizations",
        "5 · Binning",
        "6 · Feature Selection",
        "7 · Validation",
        "8 · Export",
    ]
    step = st.radio("Workflow Steps", steps, index=st.session_state.step - 1)
    st.session_state.step = steps.index(step) + 1

    st.divider()
    if st.session_state.raw_df is not None:
        st.success(f"✅ Data loaded: {st.session_state.raw_df.shape[0]:,} rows × {st.session_state.raw_df.shape[1]} cols")
    else:
        st.info("Upload a CSV to begin")

    if st.button("🔄 Reset Session", use_container_width=True):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">AutoML Studio for Binary Classification</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">End-to-end data preparation, feature engineering, validation & export for Predictions Studio</p>',
    unsafe_allow_html=True,
)
st.divider()

current_step = st.session_state.step

# ===========================================================================
# STEP 1 — Upload & Domain
# ===========================================================================
if current_step == 1:
    st.header("📂 Upload Data & Select Business Domain")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload CSV")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload your business dataset")

        if uploaded:
            try:
                df = load_csv(uploaded)
                st.session_state.raw_df = df
                st.success(f"Loaded **{df.shape[0]:,}** rows and **{df.shape[1]}** columns")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as exc:
                st.error(f"Failed to load CSV: {exc}")

    with col2:
        st.subheader("Business Problem Domain")
        domain = st.selectbox(
            "Select your industry / use case",
            list(PROBLEM_DOMAINS.keys()),
            index=list(PROBLEM_DOMAINS.keys()).index(st.session_state.domain),
        )
        st.session_state.domain = domain
        info = PROBLEM_DOMAINS[domain]
        st.info(f"**{domain}** — {info['description']}")
        st.markdown("**Common target variables:**")
        st.write(", ".join(f"`{t}`" for t in info["common_targets"]))
        st.markdown("**Recommended metrics:**")
        st.write(", ".join(info["key_metrics"])

    if st.session_state.raw_df is not None:
        st.subheader("Dataset Overview")
        summary = get_column_summary(st.session_state.raw_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(st.session_state.raw_df):,}")
        c2.metric("Columns", len(st.session_state.raw_df.columns))
        c3.metric("Missing Cells", f"{st.session_state.raw_df.isna().sum().sum():,}")
        c4.metric("Duplicates", f"{st.session_state.raw_df.duplicated().sum():,}")
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ===========================================================================
# STEP 2 — Target & Columns
# ===========================================================================
elif current_step == 2:
    st.header("🎯 Target Variable & Column Selection")

    if st.session_state.raw_df is None:
        st.warning("Please upload a CSV in Step 1 first.")
    else:
        df = st.session_state.raw_df
        domain_info = PROBLEM_DOMAINS[st.session_state.domain]
        suggested = suggest_target_column(df, domain_info["common_targets"])

        col1, col2 = st.columns([1, 2])
        with col1:
            default_idx = df.columns.tolist().index(suggested) if suggested and suggested in df.columns else 0
            target = st.selectbox("Select Target Variable", df.columns.tolist(), index=default_idx)
            st.session_state.target = target

            target_info = detect_binary_target(df[target])
            if target_info["is_binary"]:
                st.success("✅ Valid binary target")
            else:
                st.error(f"⚠️ Target has {target_info['n_unique']} unique values — binary required")

            st.json(target_info)

        with col2:
            st.subheader("Class Distribution")
            st.plotly_chart(target_distribution_chart(df, target), use_container_width=True)

        st.divider()
        st.subheader("Keep / Drop Variables")

        mode = st.radio("Selection mode", ["Keep selected columns", "Drop selected columns"], horizontal=True)

        all_features = [c for c in df.columns if c != target]

        if mode == "Keep selected columns":
            keep = st.multiselect(
                "Columns to KEEP (target is always kept)",
                all_features,
                default=st.session_state.keep_cols or all_features,
            )
            st.session_state.keep_cols = keep
            st.session_state.drop_cols = []
            filtered = apply_column_selection(df, target, keep_columns=keep)
        else:
            drop = st.multiselect("Columns to DROP", all_features, default=st.session_state.drop_cols)
            st.session_state.drop_cols = drop
            st.session_state.keep_cols = []
            filtered = apply_column_selection(df, target, drop_columns=drop)

        st.session_state.processed_df = filtered
        c1, c2 = st.columns(2)
        c1.metric("Columns after selection", len(filtered.columns))
        c2.metric("Rows", len(filtered))
        st.dataframe(filtered.head(5), use_container_width=True)

# ===========================================================================
# STEP 3 — Data Cleaning
# ===========================================================================
elif current_step == 3:
    st.header("🧹 Data Cleaning Recommendations")

    df = st.session_state.processed_df or st.session_state.raw_df
    target = st.session_state.target

    if df is None or target is None:
        st.warning("Complete Steps 1 & 2 first.")
    else:
        recommendations = generate_cleaning_recommendations(df, target)
        st.session_state.recommendations = recommendations

        severity_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for rec in recommendations:
            icon = severity_colors.get(rec["severity"], "⚪")
            with st.expander(f"{icon} [{rec['category'].upper()}] {rec['issue']}", expanded=rec["severity"] == "high"):
                st.markdown(f"**Recommendation:** {rec['recommendation']}")
                st.caption(f"Action: `{rec.get('action', 'N/A')}`")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✨ Auto-Clean (apply safe recommendations)", use_container_width=True, type="primary"):
                cleaned, config = auto_clean(df, target)
                st.session_state.processed_df = cleaned
                st.session_state.cleaning_config = config
                st.success(f"Cleaning applied! {len(df)} → {len(cleaned)} rows")
                st.rerun()

        with col2:
            if st.button("🔧 Apply ALL recommendations", use_container_width=True):
                actionable = [r for r in recommendations if r.get("action") not in ("fix_target", None)]
                cleaned, config = apply_cleaning(df, target, actionable)
                st.session_state.processed_df = cleaned
                st.session_state.cleaning_config = config
                st.success("All recommendations applied!")
                st.rerun()

        if st.session_state.cleaning_config:
            st.subheader("Applied Cleaning Config")
            st.json(st.session_state.cleaning_config)

        if st.session_state.processed_df is not None:
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.processed_df.head(10), use_container_width=True)

# ===========================================================================
# STEP 4 — Visualizations
# ===========================================================================
elif current_step == 4:
    st.header("📊 Distributions & Visualizations")

    df = st.session_state.processed_df or st.session_state.raw_df
    target = st.session_state.target

    if df is None or target is None:
        st.warning("Complete Steps 1 & 2 first.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Target & Missing", "Numeric", "Categorical", "Correlations"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(target_distribution_chart(df, target), use_container_width=True)
            with c2:
                st.plotly_chart(missing_values_chart(df), use_container_width=True)

        with tab2:
            num_cols = get_numeric_columns(df, exclude=[target])
            if num_cols:
                selected_num = st.selectbox("Numeric column", num_cols)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(numeric_histogram(df, selected_num, target), use_container_width=True)
                with c2:
                    st.plotly_chart(box_plot_by_target(df, selected_num, target), use_container_width=True)
            else:
                st.info("No numeric columns found.")

        with tab3:
            cat_cols = get_categorical_columns(df, exclude=[target])
            if cat_cols:
                selected_cat = st.selectbox("Categorical column", cat_cols)
                st.plotly_chart(categorical_bar_chart(df, selected_cat, target), use_container_width=True)
            else:
                st.info("No categorical columns found.")

        with tab4:
            st.plotly_chart(correlation_heatmap(df, target), use_container_width=True)
            if target in df.select_dtypes(include=["number"]).columns:
                st.plotly_chart(feature_target_correlation_chart(df, target), use_container_width=True)

# ===========================================================================
# STEP 5 — Binning
# ===========================================================================
elif current_step == 5:
    st.header("📦 Feature Binning")

    df = st.session_state.processed_df or st.session_state.raw_df
    target = st.session_state.target

    if df is None or target is None:
        st.warning("Complete Steps 1 & 2 first.")
    else:
        num_cols = get_numeric_columns(df, exclude=[target])
        if not num_cols:
            st.info("No numeric columns available for binning.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                bin_cols = st.multiselect("Columns to bin", num_cols, default=num_cols[:1])
            with col2:
                method = st.selectbox("Binning method", ["equal_width", "equal_frequency"])
            with col3:
                n_bins = st.slider("Number of bins", 3, 10, 5)

            if st.button("Apply Binning", type="primary") and bin_cols:
                binned_df, bin_config = apply_binning_to_dataframe(df, bin_cols, method, n_bins)
                st.session_state.processed_df = binned_df
                st.session_state.binning_config = bin_config
                st.success(f"Binned {len(bin_cols)} column(s)")

            if st.session_state.binning_config:
                st.subheader("Binning Configuration")
                st.json(st.session_state.binning_config)

            if bin_cols and target in df.columns:
                st.subheader("Weight of Evidence (WoE) & Information Value")
                woe_col = st.selectbox("Column for WoE/IV analysis", bin_cols)
                binned_col = f"{woe_col}_binned"
                check_df = st.session_state.processed_df or df
                if binned_col in check_df.columns:
                    woe_df = woe_iv_table(check_df, binned_col, target)
                else:
                    from modules.binning import create_bins
                    temp_binned, _ = create_bins(df[woe_col], method=method, n_bins=n_bins)
                    temp_df = df[[target]].copy()
                    temp_df["temp_binned"] = temp_binned
                    woe_df = woe_iv_table(temp_df, "temp_binned", target)

                if not woe_df.empty:
                    total_iv = woe_df["total_iv"].iloc[0]
                    iv_label = (
                        "Not useful"
                        if total_iv < 0.02
                        else "Weak"
                        if total_iv < 0.1
                        else "Medium"
                        if total_iv < 0.3
                        else "Strong"
                        if total_iv < 0.5
                        else "Suspicious (check)"
                    )
                    st.metric("Total Information Value (IV)", f"{total_iv:.4f}", iv_label)
                    st.dataframe(woe_df.drop(columns=["total_iv"]), use_container_width=True, hide_index=True)

# ===========================================================================
# STEP 6 — Feature Selection
# ===========================================================================
elif current_step == 6:
    st.header("🔍 Feature Selection & Visualizations")

    df = st.session_state.processed_df or st.session_state.raw_df
    target = st.session_state.target

    if df is None or target is None:
        st.warning("Complete Steps 1 & 2 first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox("Selection method", FEATURE_SELECTION_METHODS)
        with col2:
            top_k = st.slider("Top K features", 3, min(30, len(df.columns) - 1), 10)
        with col3:
            threshold = st.slider("Correlation threshold", 0.01, 0.9, 0.1, 0.01)

        if st.button("Run Feature Selection", type="primary"):
            try:
                result = run_feature_selection(df, target, method, top_k, threshold)
                st.session_state.feature_selection_result = result
                st.session_state.selected_features = get_selected_features(result)
            except Exception as exc:
                st.error(f"Feature selection failed: {exc}")

        if st.session_state.feature_selection_result is not None:
            result = st.session_state.feature_selection_result
            st.subheader("Selection Results")
            st.dataframe(result, use_container_width=True, hide_index=True)

            score_col = next(
                (c for c in result.columns if c not in ("feature", "selected", "rank")),
                None,
            )
            if score_col:
                fig = px.bar(
                    result.sort_values(score_col, ascending=True).tail(20),
                    x=score_col,
                    y="feature",
                    orientation="h",
                    title=f"Feature Scores — {method}",
                    color="selected" if "selected" in result.columns else None,
                )
                fig.update_layout(height=max(400, len(result) * 25))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Random Forest Feature Importance")
        if st.button("Compute RF Importance"):
            try:
                rf_imp = random_forest_importance(df, target)
                fig = px.bar(
                    rf_imp.head(20).sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Random Forest Feature Importance (Top 20)",
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(rf_imp, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"RF importance failed: {exc}")

        if st.session_state.selected_features:
            st.success(f"Selected features ({len(st.session_state.selected_features)}): {', '.join(st.session_state.selected_features)}")

# ===========================================================================
# STEP 7 — Validation
# ===========================================================================
elif current_step == 7:
    st.header("✅ Data & Model Validation")

    df = st.session_state.processed_df or st.session_state.raw_df
    target = st.session_state.target
    features = st.session_state.selected_features or [
        c for c in (df.columns if df is not None else []) if c != target
    ]

    if df is None or target is None:
        st.warning("Complete Steps 1 & 2 first.")
    else:
        tab1, tab2 = st.tabs(["Data Quality Checks", "Quick Model Validation"])

        with tab1:
            if st.button("Run Data Validation", type="primary"):
                report = validate_dataset(df, target, features)
                st.session_state.validation_report = report

            if st.session_state.validation_report:
                report = st.session_state.validation_report
                c1, c2, c3 = st.columns(3)
                c1.metric("Passed", report["passed"], delta_color="normal")
                c2.metric("Warnings", report["warnings"])
                c3.metric("Failed", report["failed"])

                for check in report["checks"]:
                    icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}.get(check["status"], "❓")
                    st.markdown(f"{icon} **{check['name']}** — {check['message']}")

        with tab2:
            st.caption("Quick baseline validation using Logistic Regression & Random Forest")
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("CV folds", 3, 10, 5)

            if st.button("Run Model Validation", type="primary"):
                with st.spinner("Training models..."):
                    try:
                        mv = run_model_validation(df, target, features, test_size, cv_folds)
                        st.session_state.model_validation = mv
                    except Exception as exc:
                        st.error(f"Validation failed: {exc}")

            if st.session_state.model_validation:
                mv = st.session_state.model_validation
                st.markdown(f"**Train/Test split:** {mv['split']['train']} / {mv['split']['test']}")

                for model_name, metrics in mv["models"].items():
                    st.subheader(model_name)
                    cols = st.columns(6)
                    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "CV AUC"]
                    keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "cv_auc_mean"]
                    for col, label, key in zip(cols, labels, keys):
                        col.metric(label, metrics[key])

# ===========================================================================
# STEP 8 — Export
# ===========================================================================
elif current_step == 8:
    st.header("💾 Export for Predictions Studio")

    df = st.session_state.processed_df or st.session_state.raw_df
    target = st.session_state.target

    if df is None or target is None:
        st.warning("Complete the pipeline first.")
    else:
        features = st.session_state.selected_features or [
            c for c in df.columns if c != target and not c.endswith("_binned")
        ]

        st.subheader("Export Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Domain", st.session_state.domain)
        c2.metric("Target", target)
        c3.metric("Features", len(features))
        c4.metric("Rows", f"{len(df):,}")

        manifest = build_studio_manifest(
            domain=st.session_state.domain,
            target=target,
            features=features,
            cleaning_config=st.session_state.cleaning_config,
            binning_config=st.session_state.binning_config,
            validation_report=st.session_state.validation_report,
            feature_selection={
                "method": "session",
                "selected_features": features,
                "result": (
                    st.session_state.feature_selection_result.to_dict()
                    if st.session_state.feature_selection_result is not None
                    else {}
                ),
            },
        )

        st.subheader("Studio Manifest Preview")
        st.json(manifest)

        st.divider()
        st.subheader("Download Artifacts")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "📦 Download Full ZIP Bundle",
                data=export_zip_bundle(
                    df=df,
                    manifest=manifest,
                    feature_list=features,
                    cleaning_config=st.session_state.cleaning_config,
                    binning_config=st.session_state.binning_config,
                    validation_report=st.session_state.validation_report,
                    feature_selection_results=manifest["feature_selection"],
                    domain=st.session_state.domain,
                ),
                file_name="automl_studio_export.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )

        with col2:
            st.download_button(
                "📄 Processed Dataset (CSV)",
                data=dataframe_to_csv_bytes(df),
                file_name="processed_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col3:
            st.download_button(
                "📋 Studio Manifest (JSON)",
                data=dict_to_json_bytes(manifest),
                file_name="studio_manifest.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("**Individual artifact downloads:**")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("feature_list.json", dict_to_json_bytes(features), "feature_list.json")
        d2.download_button("cleaning_config.json", dict_to_json_bytes(st.session_state.cleaning_config), "cleaning_config.json")
        d3.download_button("binning_config.json", dict_to_json_bytes(st.session_state.binning_config), "binning_config.json")
        d4.download_button("validation_report.json", dict_to_json_bytes(st.session_state.validation_report), "validation_report.json")

        st.info(
            "The **ZIP bundle** contains everything needed to plug into **Predictions Studio**: "
            "processed data, feature list, cleaning rules, binning config, validation report, and master manifest."
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("AutoML Studio v1.0 · Binary Classification Pipeline · Ready for Predictions Studio integration")
