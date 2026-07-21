"""Executive HTML report generation."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from typing import Any


def _li(items: list[str]) -> str:
    if not items:
        return "<p class='muted'>None noted.</p>"
    return "<ul>" + "".join(f"<li>{escape(str(x))}</li>" for x in items) + "</ul>"


def build_executive_report_html(
    domain: str,
    target: str,
    profile: dict[str, Any] | None = None,
    quality_score: dict[str, Any] | None = None,
    insights: dict[str, Any] | None = None,
    selected_algorithm: dict[str, Any] | None = None,
    explanation: dict[str, Any] | None = None,
    model_validation: dict[str, Any] | None = None,
    tuning_result: dict[str, Any] | None = None,
    encoding_config: dict[str, Any] | None = None,
    scaling_config: dict[str, Any] | None = None,
) -> str:
    """Self-contained HTML executive report (no external assets)."""
    profile = profile or {}
    quality_score = quality_score or {}
    insights = insights or {}
    selected_algorithm = selected_algorithm or {}
    explanation = explanation or {}
    model_validation = model_validation or {}
    tuning_result = tuning_result or {}
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metrics = selected_algorithm.get("metrics") or {}
    metric_rows = "".join(
        f"<tr><td>{escape(k)}</td><td>{escape(str(v))}</td></tr>"
        for k, v in list(metrics.items())[:12]
        if not isinstance(v, dict)
    )
    imp_rows = "".join(
        f"<tr><td>{escape(str(r.get('feature')))}</td><td>{escape(str(r.get('importance')))}</td></tr>"
        for r in (explanation.get("importances") or [])[:12]
    )
    leaderboard = model_validation.get("leaderboard") or []
    lb_rows = "".join(
        "<tr>"
        f"<td>{escape(str(r.get('model_label')))}</td>"
        f"<td>{escape(str(r.get('balance_label')))}</td>"
        f"<td>{escape(str((r.get('metrics') or {}).get('auc_roc', '')))}</td>"
        f"<td>{escape(str((r.get('metrics') or {}).get('f1', '')))}</td>"
        "</tr>"
        for r in leaderboard[:10]
    )

    qscore = quality_score.get("score", "—")
    qgrade = quality_score.get("grade", "")
    qbreakdown = quality_score.get("breakdown") or {}
    breakdown_html = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>" for k, v in qbreakdown.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AutoML Studio Executive Report</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; color: #0f172a; margin: 0; background: #f8fafc; }}
  .wrap {{ max-width: 880px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }}
  h1 {{ font-size: 1.85rem; margin: 0 0 0.35rem; letter-spacing: -0.02em; }}
  h2 {{ font-size: 1.15rem; margin: 1.6rem 0 0.55rem; border-bottom: 2px solid #0d9488; padding-bottom: 0.25rem; }}
  .sub {{ color: #64748b; margin-bottom: 1.25rem; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.15rem; margin: 0.75rem 0; }}
  .score {{ font-size: 2.4rem; font-weight: 700; color: #0f766e; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{ text-align: left; padding: 0.45rem 0.5rem; border-bottom: 1px solid #e2e8f0; }}
  th {{ color: #475569; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  .muted {{ color: #64748b; }}
  ul {{ margin: 0.35rem 0 0.35rem 1.1rem; }}
  li {{ margin: 0.25rem 0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>AutoML Studio Executive Report</h1>
  <p class="sub">{escape(domain)} · target <strong>{escape(target)}</strong> · generated {escape(created)}</p>

  <div class="card">
    <div class="score">{escape(str(qscore))}<span style="font-size:1rem;color:#64748b"> / 100 {escape(str(qgrade))}</span></div>
    <p class="muted">Data quality score</p>
    <table><thead><tr><th>Component</th><th>Points</th></tr></thead><tbody>{breakdown_html or '<tr><td colspan="2">N/A</td></tr>'}</tbody></table>
  </div>

  <h2>Business summary</h2>
  <div class="card">
    <h3 style="margin:0 0 0.4rem;font-size:1rem;">Key findings</h3>
    {_li(insights.get("summary_bullets") or [])}
    <h3 style="margin:1rem 0 0.4rem;font-size:1rem;">Risks</h3>
    {_li(insights.get("risks") or [])}
    <h3 style="margin:1rem 0 0.4rem;font-size:1rem;">Opportunities</h3>
    {_li(insights.get("opportunities") or [])}
  </div>

  <h2>Dataset profile</h2>
  <div class="card">
    <p>Rows: <strong>{escape(str(profile.get("rows", "—")))}</strong> ·
       Columns: <strong>{escape(str(profile.get("columns", "—")))}</strong> ·
       Missing cells: <strong>{escape(str(profile.get("missing_pct", "—")))}%</strong> ·
       Duplicates: <strong>{escape(str(profile.get("duplicate_pct", "—")))}%</strong></p>
  </div>

  <h2>Selected algorithm</h2>
  <div class="card">
    <p><strong>{escape(str(selected_algorithm.get("model_label", "Not selected")))}</strong>
       + {escape(str(selected_algorithm.get("balance_label", "—")))}</p>
    <table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{metric_rows or '<tr><td colspan="2">No metrics</td></tr>'}</tbody></table>
  </div>

  <h2>Model explanation (top features)</h2>
  <div class="card">
    <p class="muted">Method: {escape(str(explanation.get("method", "—")))}</p>
    <table><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>{imp_rows or '<tr><td colspan="2">Run Explain model</td></tr>'}</tbody></table>
  </div>

  <h2>Leaderboard (top 10)</h2>
  <div class="card">
    <table>
      <thead><tr><th>Model</th><th>Balancing</th><th>AUC</th><th>F1</th></tr></thead>
      <tbody>{lb_rows or '<tr><td colspan="4">No model runs</td></tr>'}</tbody>
    </table>
  </div>

  <h2>Tuning / prep notes</h2>
  <div class="card">
    <p>Hyperparameter tuning: <strong>{escape(str(tuning_result.get("status", "not run")))}</strong>
       · best score {escape(str(tuning_result.get("best_score", "—")))}</p>
    <p>Encoding actions: {escape(str(len((encoding_config or {}).get("actions") or [])))} ·
       Scaling actions: {escape(str(len((scaling_config or {}).get("actions") or [])))}</p>
  </div>

  <p class="muted" style="margin-top:2rem;">Generated by AutoML Studio for Binary Classification.</p>
</div>
</body>
</html>
"""
    return html
