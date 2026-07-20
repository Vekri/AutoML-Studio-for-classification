"""Domain configurations and constants for AutoML Studio."""

PROBLEM_DOMAINS = {
    "Banking": {
        "description": "Credit risk, fraud detection, loan default, churn",
        "common_targets": ["default", "fraud", "churn", "approved", "bad_loan"],
        "key_metrics": ["AUC-ROC", "Gini", "KS Statistic", "Precision@K"],
    },
    "Retail": {
        "description": "Customer churn, purchase prediction, coupon redemption",
        "common_targets": ["churn", "purchase", "converted", "repeat_buyer"],
        "key_metrics": ["AUC-ROC", "F1-Score", "Lift", "Precision-Recall AUC"],
    },
    "Healthcare": {
        "description": "Disease diagnosis, readmission, treatment response",
        "common_targets": ["diagnosis", "readmission", "mortality", "response"],
        "key_metrics": ["AUC-ROC", "Sensitivity", "Specificity", "F1-Score"],
    },
    "Insurance": {
        "description": "Claim prediction, policy lapse, fraud detection",
        "common_targets": ["claim", "lapse", "fraud", "renewal"],
        "key_metrics": ["AUC-ROC", "Loss Ratio", "Gini", "Calibration"],
    },
    "Telecom": {
        "description": "Customer churn, upsell, network fault prediction",
        "common_targets": ["churn", "upsell", "fault", "complaint"],
        "key_metrics": ["AUC-ROC", "Churn Rate", "Lift", "F1-Score"],
    },
    "Marketing": {
        "description": "Campaign response, lead conversion, click prediction",
        "common_targets": ["response", "converted", "clicked", "opened"],
        "key_metrics": ["AUC-ROC", "Lift", "Conversion Rate", "ROI"],
    },
    "HR / Workforce": {
        "description": "Employee attrition, performance, hiring success",
        "common_targets": ["attrition", "left", "promoted", "high_performer"],
        "key_metrics": ["AUC-ROC", "F1-Score", "Precision", "Recall"],
    },
    "Manufacturing": {
        "description": "Defect detection, equipment failure, quality control",
        "common_targets": ["defect", "failure", "reject", "anomaly"],
        "key_metrics": ["AUC-ROC", "F1-Score", "False Positive Rate", "Recall"],
    },
    "General Binary Classification": {
        "description": "Any binary classification problem",
        "common_targets": ["target", "label", "class", "outcome"],
        "key_metrics": ["AUC-ROC", "Accuracy", "F1-Score", "Precision", "Recall"],
    },
}

IMPUTATION_STRATEGIES = {
    "numeric": ["mean", "median", "mode", "constant (0)", "knn"],
    "categorical": ["mode", "constant ('Unknown')", "drop rows"],
}

OUTLIER_METHODS = ["IQR", "Z-Score (>3)", "Isolation Forest"]

BINNING_METHODS = ["equal_width", "equal_frequency", "custom"]

FEATURE_SELECTION_METHODS = [
    "correlation_threshold",
    "mutual_information",
    "chi_square",
    "anova_f",
    "rfe",
    "lasso",
]

EXPORT_ARTIFACTS = [
    "processed_dataset.csv",
    "feature_list.json",
    "binning_config.json",
    "cleaning_config.json",
    "validation_report.json",
    "domain_config.json",
    "studio_manifest.json",
]
