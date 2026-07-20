"""AutoML Studio — Binary Classification (full workflow)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import FEATURE_SELECTION_METHODS, PROBLEM_DOMAINS
from modules.binning import apply_binning_to_dataframe, create_bins, woe_iv_table
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

st.set_page_config(
    page_title="AutoML Studio | Binary Classification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = [
    "1 · Upload & Domain",
    "2 · Target & Columns",
    "3 · Data Cleaning",
    "4 · Visualizations",
    "5 · Binning",
    "6 · Feature Selection",
    "7 · Validation",
    "8 · Export",
]

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
    "rf_importance": None,
    "validation_report": {},
    "model_validation": {},
    "recommendations": [],
    "workflow_step": 1,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def go_to_step(n: int) -> None:
    st.session_state.workflow_step = max(1, min(8, int(n)))


def active_df() -> pd.DataFrame | None:
    """Return processed data if available, else raw — never use `df or other`."""
    processed = st.session_state.get("processed_df")
    if isinstance(processed, pd.DataFrame):
        return processed
    raw = st.session_state.get("raw_df")
    if isinstance(raw, pd.DataFrame):
        return raw
    return None


def ensure_domain() -> str:
    """Keep domain as a valid string (never a DataFrame)."""
    domain = st.session_state.get("domain")
    if not isinstance(domain, str) or domain not in PROBLEM_DOMAINS:
        domain = "General Binary Classification"
        st.session_state.domain = domain
    return domain


def ensure_target() -> str | None:
    """Return target if set; otherwise try to auto-suggest from raw data."""
    target = st.session_state.get("target")
    df = active_df()
    if not isinstance(df, pd.DataFrame):
        return None
    # Never use `if target` — a Series/DataFrame would raise ambiguous truth value
    if isinstance(target, str) and target in df.columns:
        return target
    domain_info = PROBLEM_DOMAINS.get(ensure_domain(), {})
    suggested = suggest_target_column(df, domain_info.get("common_targets", []))
    if isinstance(suggested, str) and suggested in df.columns:
        st.session_state.target = suggested
        return suggested
    return None


def usable_features(df: pd.DataFrame, target: str) -> list[str]:
    selected = st.session_state.get("selected_features")
    if not isinstance(selected, list):
        selected = []
    cols = [c for c in selected if isinstance(c, str) and c in df.columns and c != target]
    if cols:
        return cols
    return [
        c
        for c in df.columns
        if c != target and not str(c).endswith("_binned") and "id" not in str(c).lower()
    ]


def render_nav(current: int) -> None:
    st.divider()
    left, mid, right = st.columns([1, 2, 1])
    with left:
        if current > 1 and st.button("← Previous", use_container_width=True, key=f"prev_{current}"):
            go_to_step(current - 1)
            st.rerun()
    with mid:
        st.markdown(
            f"<p style='text-align:center;color:#64748B;'>Step {current} of 8</p>",
            unsafe_allow_html=True,
        )
    with right:
        if current < 8 and st.button("Next →", use_container_width=True, type="primary", key=f"next_{current}"):
            go_to_step(current + 1)
            st.rerun()


# ---------------------------------------------------------------------------
# CSS — avoid breaking sidebar radio controls
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0; }
    .sub-header  { font-size: 0.95rem; color: #6B7280; margin-top: 0; }
    section[data-testid="stSidebar"] { background: #1E3A5F; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #F1F5F9 !important;
    }
    section[data-testid="stSidebar"] .stRadio label { color: #F1F5F9 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 AutoML Studio")
    st.caption("Binary Classification Pipeline")
    st.divider()

    selected_label = st.radio(
        "Workflow Steps",
        STEPS,
        index=st.session_state.workflow_step - 1,
        key="sidebar_step_radio",
    )
    new_step = STEPS.index(selected_label) + 1
    if new_step != st.session_state.workflow_step:
        st.session_state.workflow_step = new_step
        st.rerun()

    st.divider()
    if isinstance(st.session_state.get("raw_df"), pd.DataFrame):
        shape = st.session_state.raw_df.shape
        st.success(f"Data loaded: {shape[0]:,} rows × {shape[1]} cols")
        tgt = st.session_state.get("target")
        if isinstance(tgt, str) and tgt:
            st.info(f"Target: `{tgt}`")
    else:
        st.warning("Upload a CSV or use sample data")

    if st.button("🔄 Reset Session", use_container_width=True):
        for key, val in DEFAULTS.items():
            if isinstance(val, dict):
                st.session_state[key] = {}
            elif isinstance(val, list):
                st.session_state[key] = []
            else:
                st.session_state[key] = val
        st.rerun()

current_step = int(st.session_state.workflow_step)

st.markdown('<p class="main-header">AutoML Studio for Binary Classification</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload → Clean → Visualize → Bin → Select Features → Validate → Export</p>',
    unsafe_allow_html=True,
)
st.divider()

# ===========================================================================
# STEP 1 — Upload & Domain
# ===========================================================================
if current_step == 1:
    st.header("📂 Upload Data & Select Business Domain")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload CSV")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        sample_path = Path(__file__).parent / "sample_data" / "banking_loan_default.csv"
        if sample_path.exists():
            if st.button("Try Sample Data (Banking)", use_container_width=True, type="primary"):
                st.session_state.raw_df = pd.read_csv(sample_path)
                st.session_state.processed_df = None
                st.session_state.domain = "Banking"
                st.session_state.target = "default"
                st.session_state.keep_cols = []
                st.session_state.drop_cols = []
                st.session_state.cleaning_config = {}
                st.session_state.binning_config = {}
                st.session_state.selected_features = []
                st.session_state.feature_selection_result = None
                st.session_state.rf_importance = None
                st.session_state.validation_report = {}
                st.session_state.model_validation = {}
                go_to_step(2)
                st.rerun()

        if uploaded is not None:
            try:
                df = load_csv(uploaded)
                st.session_state.raw_df = df
                st.session_state.processed_df = None
                st.session_state.target = None
                st.session_state.keep_cols = []
                st.session_state.selected_features = []
                st.success(f"Loaded **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as exc:
                st.error(f"Failed to load CSV: {exc}")

    with col2:
        st.subheader("Business Problem Domain")
        domains = list(PROBLEM_DOMAINS.keys())
        current_domain = st.session_state.get("domain", domains[-1])
        if not isinstance(current_domain, str) or current_domain not in domains:
            current_domain = domains[-1]
            st.session_state.domain = current_domain
        domain = st.selectbox(
            "Select industry / use case",
            domains,
            index=domains.index(current_domain),
            key="domain_select",
        )
        st.session_state.domain = str(domain)
        info = PROBLEM_DOMAINS[str(domain)]
        st.info(f"**{domain}** — {info['description']}")
        st.markdown("**Common targets:** " + ", ".join(f"`{t}`" for t in info["common_targets"]))
        st.markdown("**Key metrics:** " + ", ".join(info["key_metrics"]))

    if isinstance(st.session_state.get("raw_df"), pd.DataFrame):
        st.subheader("Dataset Overview")
        summary = get_column_summary(st.session_state.raw_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(st.session_state.raw_df):,}")
        c2.metric("Columns", len(st.session_state.raw_df.columns))
        c3.metric("Missing Cells", f"{int(st.session_state.raw_df.isna().sum().sum()):,}")
        c4.metric("Duplicates", f"{int(st.session_state.raw_df.duplicated().sum()):,}")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    render_nav(1)

# ===========================================================================
# STEP 2 — Target & Columns
# ===========================================================================
elif current_step == 2:
    st.header("🎯 Target Variable & Column Selection")

    raw = st.session_state.get("raw_df")
    if not isinstance(raw, pd.DataFrame):
        st.warning("Please upload a CSV (or sample data) in Step 1 first.")
        if st.button("Go to Step 1"):
            go_to_step(1)
            st.rerun()
    else:
        df = raw
        domain_info = PROBLEM_DOMAINS[ensure_domain()]
        suggested = suggest_target_column(df, domain_info["common_targets"])
        cols = df.columns.tolist()

        current_target = st.session_state.get("target")
        if isinstance(current_target, str) and current_target in cols:
            default_idx = cols.index(current_target)
        elif isinstance(suggested, str) and suggested in cols:
            default_idx = cols.index(suggested)
        else:
            default_idx = 0

        col1, col2 = st.columns([1, 2])
        with col1:
            target = st.selectbox(
                "Select Target Variable",
                cols,
                index=default_idx,
                key="target_select",
            )
            st.session_state.target = str(target)
            target_info = detect_binary_target(df[target])
            if target_info["is_binary"]:
                st.success("✅ Valid binary target")
            else:
                st.error(f"⚠️ Target has {target_info['n_unique']} unique values — need exactly 2")
            st.json(target_info)

        with col2:
            st.subheader("Class Distribution")
            st.plotly_chart(target_distribution_chart(df, target), use_container_width=True)

        st.divider()
        st.subheader("Keep / Drop Variables")
        mode = st.radio(
            "Selection mode",
            ["Keep selected columns", "Drop selected columns"],
            horizontal=True,
            key="col_mode",
        )
        all_features = [c for c in cols if c != target]

        keep_cols = st.session_state.get("keep_cols")
        if not isinstance(keep_cols, list):
            keep_cols = []
        drop_cols = st.session_state.get("drop_cols")
        if not isinstance(drop_cols, list):
            drop_cols = []

        if mode == "Keep selected columns":
            default_keep = [c for c in (keep_cols if keep_cols else all_features) if c in all_features]
            keep = st.multiselect("Columns to KEEP (target always kept)", all_features, default=default_keep)
            st.session_state.keep_cols = keep
            st.session_state.drop_cols = []
            filtered = apply_column_selection(df, target, keep_columns=keep)
        else:
            default_drop = [c for c in drop_cols if c in all_features]
            drop = st.multiselect("Columns to DROP", all_features, default=default_drop)
            st.session_state.drop_cols = drop
            st.session_state.keep_cols = []
            filtered = apply_column_selection(df, target, drop_columns=drop)

        st.session_state.processed_df = filtered
        c1, c2 = st.columns(2)
        c1.metric("Columns after selection", len(filtered.columns))
        c2.metric("Rows", len(filtered))
        st.dataframe(filtered.head(5), use_container_width=True)

    render_nav(2)

# ===========================================================================
# STEP 3 — Data Cleaning
# ===========================================================================
elif current_step == 3:
    st.header("🧹 Data Cleaning Recommendations")
    df = active_df()
    target = ensure_target()

    if not isinstance(df, pd.DataFrame) or not isinstance(target, str) or target not in df.columns:
        st.warning("Complete Steps 1 & 2 first (upload data and select target).")
        if st.button("Go to Step 2", key="clean_goto_2"):
            go_to_step(2)
            st.rerun()
    else:
        try:
            recommendations = generate_cleaning_recommendations(df, target)
        except Exception as exc:
            st.error(f"Could not generate cleaning recommendations: {exc}")
            recommendations = []
        st.session_state.recommendations = recommendations

        if len(recommendations) == 0:
            st.success("No major cleaning issues detected.")
        else:
            severity_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            for i, rec in enumerate(recommendations):
                icon = severity_colors.get(rec["severity"], "⚪")
                with st.expander(
                    f"{icon} [{rec['category'].upper()}] {rec['issue']}",
                    expanded=(rec.get("severity") == "high"),
                ):
                    st.markdown(f"**Recommendation:** {rec['recommendation']}")
                    st.caption(f"Action: `{rec.get('action', 'N/A')}`")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✨ Auto-Clean (safe)", use_container_width=True, type="primary"):
                try:
                    cleaned, config = auto_clean(df, target)
                    st.session_state.processed_df = cleaned
                    st.session_state.cleaning_config = config
                    st.success(
                        f"Cleaning applied! {len(df)} → {len(cleaned)} rows, {len(cleaned.columns)} cols"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Auto-clean failed: {exc}")
        with col2:
            if st.button("🔧 Apply ALL recommendations", use_container_width=True):
                try:
                    actionable = [
                        r for r in recommendations if r.get("action") not in ("fix_target", None)
                    ]
                    cleaned, config = apply_cleaning(df, target, actionable)
                    st.session_state.processed_df = cleaned
                    st.session_state.cleaning_config = config
                    st.success("All recommendations applied!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Apply cleaning failed: {exc}")

        cleaning_config = st.session_state.get("cleaning_config")
        if isinstance(cleaning_config, dict) and len(cleaning_config) > 0:
            st.subheader("Applied Cleaning Config")
            st.json(cleaning_config)

        processed = st.session_state.get("processed_df")
        if isinstance(processed, pd.DataFrame):
            st.subheader("Current Data Preview")
            st.dataframe(processed.head(10), use_container_width=True)

    render_nav(3)

# ===========================================================================
# STEP 4 — Visualizations
# ===========================================================================
elif current_step == 4:
    st.header("📊 Distributions & Visualizations")
    df = active_df()
    target = ensure_target()

    if not isinstance(df, pd.DataFrame) or not isinstance(target, str) or target not in df.columns:
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
                selected_num = st.selectbox("Numeric column", num_cols, key="viz_num")
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
                selected_cat = st.selectbox("Categorical column", cat_cols, key="viz_cat")
                st.plotly_chart(categorical_bar_chart(df, selected_cat, target), use_container_width=True)
            else:
                st.info("No categorical columns found.")

        with tab4:
            try:
                st.plotly_chart(correlation_heatmap(df, target), use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not build correlation heatmap: {exc}")
            if target in df.select_dtypes(include=["number"]).columns:
                st.plotly_chart(feature_target_correlation_chart(df, target), use_container_width=True)

    render_nav(4)

# ===========================================================================
# STEP 5 — Binning
# ===========================================================================
elif current_step == 5:
    st.header("📦 Feature Binning")
    df = active_df()
    target = ensure_target()

    if not isinstance(df, pd.DataFrame) or not isinstance(target, str) or target not in df.columns:
        st.warning("Complete Steps 1 & 2 first.")
    else:
        num_cols = get_numeric_columns(df, exclude=[target])
        # Exclude already-binned source columns if needed — keep raw numeric
        num_cols = [c for c in num_cols if not str(c).endswith("_binned")]

        if not num_cols:
            st.info("No numeric columns available for binning.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                bin_cols = st.multiselect("Columns to bin", num_cols, default=num_cols[: min(2, len(num_cols))])
            with col2:
                method = st.selectbox("Binning method", ["equal_width", "equal_frequency"])
            with col3:
                n_bins = st.slider("Number of bins", 3, 10, 5)

            if st.button("Apply Binning", type="primary") and bin_cols:
                binned_df, bin_config = apply_binning_to_dataframe(df, bin_cols, method, n_bins)
                st.session_state.processed_df = binned_df
                st.session_state.binning_config = bin_config
                st.success(f"Binned {len(bin_cols)} column(s)")
                st.rerun()

            if st.session_state.binning_config:
                st.subheader("Binning Configuration")
                st.json(st.session_state.binning_config)

            if bin_cols:
                st.subheader("Weight of Evidence (WoE) & Information Value")
                woe_col = st.selectbox("Column for WoE/IV", bin_cols, key="woe_col")
                check_df = active_df()
                if check_df is None:
                    check_df = df
                binned_col = f"{woe_col}_binned"
                try:
                    if binned_col in check_df.columns:
                        woe_df = woe_iv_table(check_df, binned_col, target)
                    else:
                        temp_binned, _ = create_bins(df[woe_col], method=method, n_bins=n_bins)
                        temp_df = df[[target]].copy()
                        temp_df["temp_binned"] = temp_binned
                        woe_df = woe_iv_table(temp_df, "temp_binned", target)

                    if woe_df is not None and not woe_df.empty:
                        total_iv = float(woe_df["total_iv"].iloc[0])
                        iv_label = (
                            "Not useful" if total_iv < 0.02
                            else "Weak" if total_iv < 0.1
                            else "Medium" if total_iv < 0.3
                            else "Strong" if total_iv < 0.5
                            else "Suspicious (check)"
                        )
                        st.metric("Total Information Value (IV)", f"{total_iv:.4f}", iv_label)
                        st.dataframe(woe_df.drop(columns=["total_iv"]), use_container_width=True, hide_index=True)
                    else:
                        st.info("WoE/IV needs a binary target with both classes present.")
                except Exception as exc:
                    st.warning(f"WoE/IV could not be computed: {exc}")

    render_nav(5)

# ===========================================================================
# STEP 6 — Feature Selection
# ===========================================================================
elif current_step == 6:
    st.header("🔍 Feature Selection & Visualizations")
    df = active_df()
    target = ensure_target()

    if not isinstance(df, pd.DataFrame) or not isinstance(target, str) or target not in df.columns:
        st.warning("Complete Steps 1 & 2 first.")
    else:
        n_feat = max(1, len(df.columns) - 1)
        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox("Selection method", FEATURE_SELECTION_METHODS)
        with col2:
            top_k = st.slider("Top K features", 1, min(30, n_feat), min(10, n_feat))
        with col3:
            threshold = st.slider("Correlation threshold", 0.01, 0.9, 0.1, 0.01)

        if st.button("Run Feature Selection", type="primary"):
            with st.spinner("Selecting features..."):
                try:
                    result = run_feature_selection(df, target, method, top_k, threshold)
                    if result is None or result.empty:
                        st.error("No features scored. Check that numeric/usable features exist.")
                    else:
                        st.session_state.feature_selection_result = result
                        st.session_state.selected_features = get_selected_features(result)
                        st.success(f"Selected {len(st.session_state.selected_features)} features")
                except Exception as exc:
                    st.error(f"Feature selection failed: {exc}")

        if st.session_state.feature_selection_result is not None:
            result = st.session_state.feature_selection_result
            st.subheader("Selection Results")
            st.dataframe(result, use_container_width=True, hide_index=True)
            score_col = next((c for c in result.columns if c not in ("feature", "selected", "rank")), None)
            if score_col:
                plot_df = result.sort_values(score_col, ascending=True).tail(20)
                fig = px.bar(
                    plot_df,
                    x=score_col,
                    y="feature",
                    orientation="h",
                    title=f"Feature Scores — {method}",
                    color="selected" if "selected" in result.columns else None,
                )
                fig.update_layout(height=max(400, len(plot_df) * 28))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Random Forest Feature Importance")
        if st.button("Compute RF Importance"):
            with st.spinner("Training Random Forest..."):
                try:
                    st.session_state.rf_importance = random_forest_importance(df, target)
                except Exception as exc:
                    st.error(f"RF importance failed: {exc}")

        if st.session_state.rf_importance is not None:
            rf_imp = st.session_state.rf_importance
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

        if st.session_state.selected_features:
            st.success(
                f"Selected features ({len(st.session_state.selected_features)}): "
                + ", ".join(st.session_state.selected_features)
            )

    render_nav(6)

# ===========================================================================
# STEP 7 — Validation
# ===========================================================================
elif current_step == 7:
    st.header("✅ Data & Model Validation")
    df = active_df()
    target = ensure_target()

    if not isinstance(df, pd.DataFrame) or not isinstance(target, str) or target not in df.columns:
        st.warning("Complete Steps 1 & 2 first.")
    else:
        features = usable_features(df, target)
        st.caption(f"Using {len(features)} features for validation")

        tab1, tab2 = st.tabs(["Data Quality Checks", "Quick Model Validation"])

        with tab1:
            if st.button("Run Data Validation", type="primary"):
                st.session_state.validation_report = validate_dataset(df, target, features)

            report = st.session_state.validation_report
            if report and report.get("checks"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Passed", report.get("passed", 0))
                c2.metric("Warnings", report.get("warnings", 0))
                c3.metric("Failed", report.get("failed", 0))
                for check in report["checks"]:
                    icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}.get(check["status"], "❓")
                    st.markdown(f"{icon} **{check['name']}** — {check['message']}")

        with tab2:
            st.caption("Baseline models: Logistic Regression & Random Forest")
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("CV folds", 2, 10, 5)

            if st.button("Run Model Validation", type="primary"):
                with st.spinner("Training models..."):
                    try:
                        st.session_state.model_validation = run_model_validation(
                            df, target, features, test_size, cv_folds
                        )
                    except Exception as exc:
                        st.error(f"Validation failed: {exc}")

            mv = st.session_state.model_validation
            if mv and mv.get("models"):
                st.markdown(f"**Train/Test split:** {mv['split']['train']} / {mv['split']['test']}")
                if mv.get("features_used"):
                    st.caption("Features: " + ", ".join(mv["features_used"]))
                for model_name, metrics in mv["models"].items():
                    st.subheader(model_name)
                    cols = st.columns(6)
                    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "CV AUC"]
                    keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "cv_auc_mean"]
                    for col, label, key in zip(cols, labels, keys):
                        val = metrics.get(key)
                        col.metric(label, val if val is not None else "N/A")

    render_nav(7)

# ===========================================================================
# STEP 8 — Export
# ===========================================================================
elif current_step == 8:
    st.header("💾 Export for Predictions Studio")
    df = active_df()
    target = ensure_target()

    if not isinstance(df, pd.DataFrame) or not isinstance(target, str) or target not in df.columns:
        st.warning("Complete the pipeline first (at least Steps 1 & 2).")
    else:
        features = usable_features(df, target)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Domain", st.session_state.domain)
        c2.metric("Target", target)
        c3.metric("Features", len(features))
        c4.metric("Rows", f"{len(df):,}")

        fs_result = {}
        if st.session_state.feature_selection_result is not None:
            try:
                fs_result = st.session_state.feature_selection_result.to_dict(orient="records")
            except Exception:
                fs_result = {}

        manifest = build_studio_manifest(
            domain=st.session_state.domain,
            target=target,
            features=features,
            cleaning_config=st.session_state.cleaning_config or {},
            binning_config=st.session_state.binning_config or {},
            validation_report=st.session_state.validation_report or {},
            feature_selection={
                "method": "session",
                "selected_features": features,
                "result": fs_result,
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
                    cleaning_config=st.session_state.cleaning_config or {},
                    binning_config=st.session_state.binning_config or {},
                    validation_report=st.session_state.validation_report or {},
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

        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("feature_list.json", dict_to_json_bytes(features), "feature_list.json")
        d2.download_button(
            "cleaning_config.json",
            dict_to_json_bytes(st.session_state.cleaning_config or {}),
            "cleaning_config.json",
        )
        d3.download_button(
            "binning_config.json",
            dict_to_json_bytes(st.session_state.binning_config or {}),
            "binning_config.json",
        )
        d4.download_button(
            "validation_report.json",
            dict_to_json_bytes(st.session_state.validation_report or {}),
            "validation_report.json",
        )

        st.info(
            "ZIP bundle includes processed data, feature list, cleaning/binning configs, "
            "validation report, and studio_manifest.json for Predictions Studio."
        )

    render_nav(8)

st.divider()
st.caption("AutoML Studio v1.1 · Free & Open Source · MIT License")
