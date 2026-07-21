"""Business insights from profile, features, metrics, and domain context."""

from __future__ import annotations

from typing import Any


def generate_business_insights(
    domain: str,
    target: str,
    profile: dict[str, Any] | None = None,
    quality_score: dict[str, Any] | None = None,
    feature_selection: list[dict[str, Any]] | None = None,
    selected_algorithm: dict[str, Any] | None = None,
    explanation: dict[str, Any] | None = None,
    model_validation: dict[str, Any] | None = None,
    binning_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Domain-templated narrative bullets for executives."""
    bullets: list[str] = []
    risks: list[str] = []
    opportunities: list[str] = []

    profile = profile or {}
    quality_score = quality_score or {}
    selected_algorithm = selected_algorithm or {}
    explanation = explanation or {}
    model_validation = model_validation or {}

    rows = profile.get("rows")
    if rows:
        bullets.append(f"Analyzed {rows:,} records for domain “{domain}” with target “{target}”.")

    score = quality_score.get("score")
    if score is not None:
        grade = quality_score.get("grade", "")
        bullets.append(f"Data quality score is {score}/100 ({grade}).")
        if score < 70:
            risks.append("Data quality is below 70 — prioritize missing values, duplicates, and outliers before production.")

    tgt = profile.get("target") or {}
    if tgt.get("class_counts"):
        counts = tgt["class_counts"]
        parts = ", ".join(f"{k}: {v}" for k, v in list(counts.items())[:4])
        bullets.append(f"Class distribution — {parts}.")
        vals = list(counts.values())
        if len(vals) == 2 and min(vals) / max(vals) < 0.2:
            risks.append("Severe class imbalance — use balancing (SMOTE / class weights) and track recall/precision, not only accuracy.")
            opportunities.append("Imbalance-aware models can improve minority-class detection (defaults, fraud, churn, etc.).")

    top_feats: list[str] = []
    if explanation.get("top_features"):
        top_feats = list(explanation["top_features"])[:5]
    elif feature_selection:
        ranked = sorted(
            feature_selection,
            key=lambda r: float(r.get("score") or r.get("importance") or 0),
            reverse=True,
        )
        top_feats = [str(r.get("feature") or r.get("column")) for r in ranked[:5]]

    if top_feats:
        bullets.append("Top drivers: " + ", ".join(top_feats) + ".")
        opportunities.append(
            f"Focus monitoring and policy rules on leading drivers ({', '.join(top_feats[:3])})."
        )

    algo_label = selected_algorithm.get("model_label")
    bal_label = selected_algorithm.get("balance_label")
    metrics = selected_algorithm.get("metrics") or {}
    if algo_label:
        line = f"Recommended model: {algo_label}"
        if bal_label:
            line += f" with {bal_label}"
        if metrics.get("auc_roc") is not None:
            line += f" (AUC {metrics.get('auc_roc')}, F1 {metrics.get('f1')})"
        bullets.append(line + ".")

    if metrics.get("recall") is not None and float(metrics["recall"]) < 0.5:
        risks.append("Recall is below 0.50 — the model may miss many positive cases; consider threshold tuning or recall-oriented ranking.")
    if metrics.get("precision") is not None and float(metrics["precision"]) < 0.5:
        risks.append("Precision is below 0.50 — false positives may be high; review operating threshold with business cost.")

    if binning_config:
        opportunities.append("WoE/IV binning is available for interpretable scorecard-style features in regulated domains.")

    best = model_validation.get("best") or {}
    if best.get("model_label") and best.get("model_label") != algo_label:
        bullets.append(f"Leaderboard leader during comparison: {best.get('model_label')}.")

    domain_l = (domain or "").lower()
    if "bank" in domain_l or "credit" in domain_l or "loan" in domain_l:
        opportunities.append("For credit risk, validate stability of top drivers across time and document adverse-action reasons.")
    elif "churn" in domain_l or "telecom" in domain_l or "retail" in domain_l:
        opportunities.append("For retention, pair model scores with treatment eligibility to estimate lift of outreach campaigns.")
    elif "health" in domain_l:
        opportunities.append("For healthcare use-cases, confirm PHI handling and clinically review top drivers before deployment.")
    else:
        opportunities.append("Export the package to Predictions Studio and monitor AUC/F1 drift on fresh batches.")

    if not bullets:
        bullets.append("Run profiling, cleaning, feature selection, and model comparison to generate insights.")

    return {
        "domain": domain,
        "target": target,
        "summary_bullets": bullets,
        "risks": risks,
        "opportunities": opportunities,
        "top_features": top_feats,
        "recommended_algorithm": {
            "model_label": algo_label,
            "balance_label": bal_label,
            "metrics": metrics,
        },
        "quality_score": score,
    }
