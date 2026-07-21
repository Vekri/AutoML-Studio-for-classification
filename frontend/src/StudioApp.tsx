import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, BalanceMethod, DomainInfo, ModelOption, SessionPayload } from "./api";

const STEPS = [
  { id: 1, title: "Upload & Domain", blurb: "CSV + business problem" },
  { id: 2, title: "Target & Columns", blurb: "Keep / drop variables" },
  { id: 3, title: "Data Cleaning", blurb: "Recommendations" },
  { id: 4, title: "Visualizations", blurb: "Distributions" },
  { id: 5, title: "Binning", blurb: "WoE / IV" },
  { id: 6, title: "Feature Selection", blurb: "Rank & pick" },
  { id: 7, title: "Balance & Models", blurb: "Sampling + ML models" },
  { id: 8, title: "Export", blurb: "Predictions Studio" },
];

function PreviewTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return <p className="muted">No preview rows</p>;
  const cols = Object.keys(rows[0]);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c}>{String(r[c] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function StudioApp({
  embedded = false,
  initialStep = 1,
  projectId,
  projectName,
  domain: domainProp,
  loadSampleOnStart = false,
}: {
  embedded?: boolean;
  initialStep?: number;
  projectId?: string;
  projectName?: string;
  domain?: string;
  loadSampleOnStart?: boolean;
} = {}) {
  const [step, setStep] = useState(initialStep);
  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [methods, setMethods] = useState<string[]>([]);
  const [balanceOptions, setBalanceOptions] = useState<BalanceMethod[]>([]);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [domain, setDomain] = useState(domainProp || "Banking");
  const [session, setSession] = useState<SessionPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  // Step 2
  const [target, setTarget] = useState("default");
  const [mode, setMode] = useState<"keep" | "drop">("keep");
  const [selectedCols, setSelectedCols] = useState<string[]>([]);
  const [targetDist, setTargetDist] = useState<{ class: string; count: number }[]>([]);
  const [targetInfo, setTargetInfo] = useState<Record<string, unknown> | null>(null);

  // Step 3
  const [recs, setRecs] = useState<Record<string, unknown>[]>([]);
  const [cleanTab, setCleanTab] = useState<"missing" | "outliers" | "duplicates" | "encoding" | "scaling">(
    "missing"
  );
  const [outlierMethod, setOutlierMethod] = useState("iqr");
  const [outlierReport, setOutlierReport] = useState<Record<string, unknown> | null>(null);
  const [encodingRecs, setEncodingRecs] = useState<Record<string, unknown> | null>(null);
  const [scalingRecs, setScalingRecs] = useState<Record<string, unknown> | null>(null);
  const [qualityScore, setQualityScore] = useState<Record<string, unknown> | null>(null);
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null);

  // Step 4
  const [viz, setViz] = useState<Awaited<ReturnType<typeof api.visualizations>> | null>(null);
  const [vizTab, setVizTab] = useState("target");
  const [numCol, setNumCol] = useState("");
  const [catCol, setCatCol] = useState("");
  const [catData, setCatData] = useState<Record<string, unknown>[]>([]);

  // Step 5
  const [binCols, setBinCols] = useState<string[]>([]);
  const [binMethod, setBinMethod] = useState("equal_width");
  const [nBins, setNBins] = useState(5);
  const [woe, setWoe] = useState<Record<string, Record<string, unknown>[]> | null>(null);

  // Step 6
  const [fsMethod, setFsMethod] = useState("mutual_information");
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.1);
  const [fsResult, setFsResult] = useState<Record<string, unknown>[]>([]);
  const [rfResult, setRfResult] = useState<Record<string, unknown>[]>([]);

  // Step 7
  const [dataReport, setDataReport] = useState<Record<string, unknown> | null>(null);
  const [modelResult, setModelResult] = useState<Record<string, unknown> | null>(null);
  const [testSize, setTestSize] = useState(0.2);
  const [cvFolds, setCvFolds] = useState(5);
  const [selectedBalances, setSelectedBalances] = useState<string[]>([
    "none",
    "class_weight",
    "smote",
  ]);
  const [selectedModels, setSelectedModels] = useState<string[]>([
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
  ]);
  const [runAllCombos, setRunAllCombos] = useState(true);
  const [selectionMetric, setSelectionMetric] = useState("auc_roc");
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Record<string, unknown> | null>(null);
  const [tuningResult, setTuningResult] = useState<Record<string, unknown> | null>(null);
  const [explanation, setExplanation] = useState<Record<string, unknown> | null>(null);
  const [reduction, setReduction] = useState<Record<string, unknown> | null>(null);
  const [feConfig, setFeConfig] = useState<Record<string, unknown> | null>(null);

  // Step 8
  const [manifest, setManifest] = useState<Record<string, unknown> | null>(null);
  const [insights, setInsights] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    api
      .domains()
      .then((d) => {
        setDomains(d.domains);
        setMethods(d.feature_selection_methods);
        setBalanceOptions(d.balance_methods || []);
        setModelOptions(d.models || []);
        if (d.feature_selection_methods.length)
          setFsMethod(d.feature_selection_methods[1] || d.feature_selection_methods[0]);
        if (d.balance_methods?.length) {
          setSelectedBalances(
            d.balance_methods
              .map((b) => b.id)
              .filter((id) => ["none", "class_weight", "smote", "random_oversample"].includes(id))
          );
        }
        if (d.models?.length) {
          setSelectedModels(
            d.models
              .map((m) => m.id)
              .filter((id) =>
                ["logistic_regression", "random_forest", "gradient_boosting", "decision_tree"].includes(
                  id
                )
              )
          );
        }
      })
      .catch((e) => setError(String(e.message || e)));
  }, []);

  useEffect(() => {
    setStep(initialStep);
  }, [initialStep]);

  useEffect(() => {
    if (domainProp) setDomain(domainProp);
  }, [domainProp]);

  const featureOptions = useMemo(() => {
    if (!session) return [];
    return session.columns.filter((c) => c !== target);
  }, [session, target]);

  async function run<T>(fn: () => Promise<T>, okMsg?: string): Promise<T | null> {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await fn();
      if (okMsg) setMessage(okMsg);
      return result;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function loadSample() {
    const data = await run(() => api.sample(domain), "Sample banking dataset loaded");
    if (!data) return;
    setSession(data);
    setProfile(data.profile || null);
    setQualityScore(data.quality_score || null);
    setTarget(data.suggested_target || data.target || "default");
    setSelectedCols(data.columns.filter((c) => c !== (data.suggested_target || "default") && !c.toLowerCase().includes("id")));
    if (!embedded) setStep(2);
  }

  useEffect(() => {
    if (!loadSampleOnStart || session) return;
    loadSample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadSampleOnStart, projectId]);

  async function onUpload(file: File | null) {
    if (!file) return;
    const data = await run(() => api.upload(file, domain), "CSV uploaded");
    if (!data) return;
    setSession(data);
    setProfile(data.profile || null);
    setQualityScore(data.quality_score || null);
    const t = data.suggested_target || data.columns[data.columns.length - 1];
    setTarget(t || "");
    setSelectedCols(data.columns.filter((c) => c !== t));
  }

  async function refreshTarget(t: string) {
    if (!session) return;
    setTarget(t);
    const info = await run(() => api.targetInfo(session.session_id, t));
    if (!info) return;
    setTargetInfo(info.info);
    setTargetDist(info.distribution);
  }

  useEffect(() => {
    if (session && step === 2 && target) {
      refreshTarget(target);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.session_id, step]);

  async function saveColumns() {
    if (!session) return;
    const data = await run(
      () =>
        api.selectColumns(session.session_id, {
          target,
          mode,
          columns: selectedCols,
        }),
      "Columns saved"
    );
    if (data) {
      setSession(data);
      setStep(3);
    }
  }

  async function loadCleaning() {
    if (!session) return;
    const data = await run(() => api.cleaning(session.session_id));
    if (data) setRecs(data.recommendations);
  }

  useEffect(() => {
    if (step === 3 && session) loadCleaning();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, session?.session_id]);

  async function applyClean(modeClean: "auto" | "all") {
    if (!session) return;
    const data = await run(() => api.applyCleaning(session.session_id, modeClean), "Cleaning applied");
    if (data) {
      setSession(data);
      setQualityScore(data.quality_score || null);
      setProfile(data.profile || null);
      await loadCleaning();
    }
  }

  async function runOutliers(mode: "detect" | "cap" | "drop_rows") {
    if (!session) return;
    const data = await run(
      () => api.outliers(session.session_id, { method: outlierMethod, mode }),
      mode === "detect" ? "Outliers detected" : "Outlier treatment applied"
    );
    if (data) {
      setOutlierReport(data.report);
      setSession(data);
      if (data.quality_score) setQualityScore(data.quality_score);
    }
  }

  async function loadEncoding(apply = false) {
    if (!session) return;
    const data = await run(
      () => api.encoding(session.session_id, apply),
      apply ? "Encoding applied" : "Encoding recommendations ready"
    );
    if (data) {
      setEncodingRecs(data.recommendations);
      setSession(data);
    }
  }

  async function loadScaling(apply = false) {
    if (!session) return;
    const data = await run(
      () => api.scaling(session.session_id, apply),
      apply ? "Scaling applied" : "Scaling recommendations ready"
    );
    if (data) {
      setScalingRecs(data.recommendations);
      setSession(data);
    }
  }

  async function runFeatureEng() {
    if (!session) return;
    const data = await run(() => api.featureEng(session.session_id), "Features engineered");
    if (data) {
      setFeConfig(data.config);
      setSession(data);
    }
  }

  async function runReduce(applyDrops = false) {
    if (!session) return;
    const data = await run(
      () =>
        api.reduce(session.session_id, {
          corr_threshold: 0.92,
          run_pca: true,
          apply_drops: applyDrops,
        }),
      applyDrops ? "Correlated variables dropped" : "Reduction analysis ready"
    );
    if (data) {
      setReduction(data.reduction);
      setSession(data);
    }
  }

  async function runTune() {
    if (!session) return;
    const data = await run(
      () => api.tune(session.session_id, { selection_metric: selectionMetric }),
      "Hyperparameter tuning complete"
    );
    if (data) {
      setTuningResult(data.tuning);
      if (data.selected_algorithm) setSelectedAlgorithm(data.selected_algorithm);
    }
  }

  async function runExplain() {
    if (!session) return;
    const data = await run(() => api.explain(session.session_id), "Model explanation ready");
    if (data) setExplanation(data.explanation);
  }

  async function loadInsights() {
    if (!session) return;
    const data = await run(() => api.insights(session.session_id));
    if (data) setInsights(data.insights);
  }

  async function loadViz() {
    if (!session) return;
    const data = await run(() => api.visualizations(session.session_id));
    if (!data) return;
    setViz(data);
    setNumCol(data.numeric_columns[0] || "");
    setCatCol(data.categorical_columns[0] || "");
    setBinCols(data.numeric_columns.slice(0, 2));
  }

  useEffect(() => {
    if ((step === 4 || step === 5) && session) loadViz();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, session?.session_id]);

  useEffect(() => {
    if (!session || !catCol || step !== 4) return;
    api.vizCategorical(session.session_id, catCol).then((d) => setCatData(d.grouped)).catch(() => setCatData([]));
  }, [session, catCol, step]);

  async function applyBinning() {
    if (!session) return;
    const data = await run(
      () => api.binning(session.session_id, { columns: binCols, method: binMethod, n_bins: nBins }),
      "Binning applied"
    );
    if (data) {
      setSession(data);
      setWoe(data.woe_tables || null);
    }
  }

  async function runFeatures() {
    if (!session) return;
    const data = await run(
      () => api.features(session.session_id, { method: fsMethod, top_k: topK, threshold }),
      "Feature selection complete"
    );
    if (data) {
      setSession(data);
      setFsResult(data.result || []);
    }
  }

  async function runRf() {
    if (!session) return;
    const data = await run(() => api.rfImportance(session.session_id));
    if (data) setRfResult(data.result);
  }

  async function runDataValidation() {
    if (!session) return;
    const data = await run(() => api.validateData(session.session_id), "Data validation done");
    if (data) setDataReport(data.report);
  }

  async function runModelValidation() {
    if (!session) return;
    if (!selectedModels.length) {
      setError("Select at least one model");
      return;
    }
    if (!selectedBalances.length) {
      setError("Select at least one balancing method");
      return;
    }
    const data = await run(
      () =>
        api.validateModels(session.session_id, {
          test_size: testSize,
          cv_folds: cvFolds,
          models: selectedModels,
          balance_methods: selectedBalances,
          run_all_combinations: runAllCombos,
          selection_metric: selectionMetric,
          auto_select_best: true,
        }),
      "Model suite complete — best algorithm selected"
    );
    if (data) {
      setModelResult(data.result);
      if (data.selected_algorithm) setSelectedAlgorithm(data.selected_algorithm);
      else if (data.result.selected_algorithm)
        setSelectedAlgorithm(data.result.selected_algorithm as Record<string, unknown>);
    }
  }

  async function pickBestAlgorithm() {
    if (!session) return;
    const data = await run(
      () => api.autoSelectAlgorithm(session.session_id, selectionMetric),
      "Best algorithm selected"
    );
    if (data) setSelectedAlgorithm(data.selected_algorithm);
  }

  async function pickAlgorithm(modelId: string, balanceMethod: string) {
    if (!session) return;
    const data = await run(
      () =>
        api.selectAlgorithm(session.session_id, {
          model_id: modelId,
          balance_method: balanceMethod,
          selection_metric: selectionMetric,
        }),
      "Algorithm selected"
    );
    if (data) setSelectedAlgorithm(data.selected_algorithm);
  }

  async function loadManifest() {
    if (!session) return;
    const data = await run(() => api.manifest(session.session_id));
    if (data) setManifest(data);
  }

  useEffect(() => {
    if (step === 8 && session) {
      loadManifest();
      loadInsights();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, session?.session_id]);

  async function resetAll() {
    if (session) await api.reset(session.session_id).catch(() => null);
    setSession(null);
    setStep(1);
    setRecs([]);
    setViz(null);
    setFsResult([]);
    setRfResult([]);
    setDataReport(null);
    setModelResult(null);
    setSelectedAlgorithm(null);
    setManifest(null);
    setMessage(null);
    setError(null);
  }

  const domainInfo = domains.find((d) => d.name === domain);

  const scoreKey = useMemo(() => {
    if (!fsResult.length) return null;
    return Object.keys(fsResult[0]).find((k) => !["feature", "selected", "rank"].includes(k)) || null;
  }, [fsResult]);

  return (
    <div className={embedded ? "dash-studio-wrap" : "app-shell"}>
      {!embedded && (
      <aside className="sidebar">
        <div className="brand">
          <Link to="/" className="brand-mark" style={{ textDecoration: "none", color: "inherit" }}>
            IP
          </Link>
          <div>
            <h1>IntelliPredict Studio</h1>
            <p>Binary classification pipeline</p>
          </div>
        </div>

        <div className="progress-mini">
          <div className="label">
            <span>Progress</span>
            <span>{Math.round((step / 8) * 100)}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${(step / 8) * 100}%` }} />
          </div>
        </div>

        <div className="step-list">
          {STEPS.map((s) => (
            <button
              key={s.id}
              className={`step-btn ${step === s.id ? "active" : ""} ${s.id < step ? "done" : ""}`}
              onClick={() => setStep(s.id)}
            >
              <span className="step-num">{s.id < step ? "✓" : s.id}</span>
              <span>
                <strong>{s.title}</strong>
                <small>{s.blurb}</small>
              </span>
            </button>
          ))}
        </div>

        <div className="sidebar-meta">
          {session ? (
            <>
              <div>
                <strong>{session.filename}</strong>
                <div>
                  {session.rows.toLocaleString()} rows · {session.columns.length} cols
                </div>
              </div>
              {session.target && (
                <div>
                  Target: <strong>{session.target}</strong>
                </div>
              )}
              <div>Domain: {session.domain}</div>
            </>
          ) : (
            <div>No dataset loaded yet</div>
          )}
          <button className="ghost" onClick={resetAll}>
            Reset session
          </button>
        </div>
      </aside>
      )}

      <main className={embedded ? "main embedded-main" : "main"} key={step}>
        {embedded && projectName && (
          <div className="dash-studio-banner">
            <h1>{projectName}</h1>
            <p className="muted">Step {step} of 8 · {STEPS[step - 1].title}</p>
          </div>
        )}
        <div className="topbar">
          <div>
            <h2>{STEPS[step - 1].title}</h2>
            <p>
              {step === 1 && "Load business data and pick the industry problem domain."}
              {step === 2 && "Choose the binary target and keep or drop predictor columns."}
              {step === 3 && "Review cleaning suggestions and apply safe fixes."}
              {step === 4 && "Explore distributions, missingness, and correlations."}
              {step === 5 && "Bin numeric features and inspect WoE / Information Value."}
              {step === 6 && "Rank features and select the strongest predictors."}
              {step === 7 && "Choose balancing methods and models — run all combinations and compare."}
              {step === 8 && "Download the package for Predictions Studio."}
            </p>
          </div>
          <div className="step-chip">
            Step <em>{step}</em> of 8
          </div>
        </div>

        {error && <div className="error">{error}</div>}
        {message && <div className="success">{message}</div>}
        {busy && <div className="success">Working…</div>}

        {step === 1 && (
          <div className="panel grid-2">
            <div>
              <p className="panel-title">Data source</p>
              <div className="field">
                <label>Business domain</label>
                <select
                  value={domain}
                  onChange={async (e) => {
                    const next = e.target.value;
                    setDomain(next);
                    if (session) {
                      const updated = await run(() => api.updateDomain(session.session_id, next));
                      if (updated) setSession(updated);
                    }
                  }}
                >
                  {domains.map((d) => (
                    <option key={d.name} value={d.name}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </div>
              {domainInfo && (
                <p className="muted" style={{ marginTop: "-0.35rem", marginBottom: "1rem" }}>
                  {domainInfo.description}
                  <br />
                  Common targets: {domainInfo.common_targets.join(", ")}
                </p>
              )}
              <div className="dropzone">
                <strong>Upload CSV</strong>
                <p>Banking, retail, healthcare, or any binary classification dataset</p>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => onUpload(e.target.files?.[0] || null)}
                />
              </div>
              <div className="btn-row">
                <button className="primary" disabled={busy} onClick={loadSample}>
                  Try sample data (Banking)
                </button>
              </div>
            </div>
            <div>
              <p className="panel-title">Data profiling</p>
              {session ? (
                <div className="split-stack">
                  {qualityScore && (
                    <div className="metric" style={{ marginBottom: "0.5rem" }}>
                      <div className="label">Data quality score</div>
                      <div className="value">
                        {String(qualityScore.score)}
                        <span style={{ fontSize: "1rem", color: "var(--muted)" }}>
                          {" "}
                          / 100 · grade {String(qualityScore.grade)}
                        </span>
                      </div>
                    </div>
                  )}
                  <div className="grid-2">
                    <div className="metric">
                      <div className="label">Rows</div>
                      <div className="value">{session.rows}</div>
                    </div>
                    <div className="metric">
                      <div className="label">Columns</div>
                      <div className="value">{session.columns.length}</div>
                    </div>
                    <div className="metric">
                      <div className="label">Missing %</div>
                      <div className="value">
                        {String(profile?.missing_pct ?? session.missing_cells ?? 0)}
                      </div>
                    </div>
                    <div className="metric">
                      <div className="label">Duplicates</div>
                      <div className="value">
                        {String(profile?.duplicate_rows ?? session.duplicates ?? 0)}
                      </div>
                    </div>
                  </div>
                  {!!(profile?.warnings as string[] | undefined)?.length && (
                    <div className="rec">
                      <h4>Profile warnings</h4>
                      {(profile?.warnings as string[]).map((w, i) => (
                        <p key={i}>{w}</p>
                      ))}
                    </div>
                  )}
                  <PreviewTable rows={session.preview} />
                </div>
              ) : (
                <div className="empty-state">
                  <strong>No data yet</strong>
                  Upload a CSV or load the banking sample to begin the pipeline.
                </div>
              )}
            </div>
          </div>
        )}

        {step === 2 && session && (
          <div className="panel">
            <div className="grid-2">
              <div>
                <div className="field">
                  <label>Target variable</label>
                  <select value={target} onChange={(e) => refreshTarget(e.target.value)}>
                    {session.columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
                {targetInfo && (
                  <div>
                    {targetInfo.is_binary ? (
                      <span className="badge ok">Valid binary target</span>
                    ) : (
                      <span className="badge danger">Not binary ({String(targetInfo.n_unique)} classes)</span>
                    )}
                    <pre className="mono" style={{ fontSize: "0.8rem", whiteSpace: "pre-wrap" }}>
                      {JSON.stringify(targetInfo, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
              <div>
                <div className="chart-box">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={targetDist}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="class" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#0d9488" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            <div className="field" style={{ marginTop: "1rem" }}>
              <label>Selection mode</label>
              <select value={mode} onChange={(e) => setMode(e.target.value as "keep" | "drop")}>
                <option value="keep">Keep selected columns</option>
                <option value="drop">Drop selected columns</option>
              </select>
            </div>
            <div className="checks">
              {featureOptions.map((c) => (
                <label key={c}>
                  <input
                    type="checkbox"
                    checked={selectedCols.includes(c)}
                    onChange={(e) => {
                      setSelectedCols((prev) =>
                        e.target.checked ? [...prev, c] : prev.filter((x) => x !== c)
                      );
                    }}
                  />
                  {c}
                </label>
              ))}
            </div>
            <div className="btn-row">
              <button className="primary" disabled={busy} onClick={saveColumns}>
                Save & continue
              </button>
            </div>
          </div>
        )}

        {step === 3 && session && (
          <div className="panel">
            {qualityScore && (
              <div className="success" style={{ marginBottom: "1rem" }}>
                Quality score: <strong>{String(qualityScore.score)}/100 ({String(qualityScore.grade)})</strong>
                {(qualityScore.issues as string[] | undefined)?.length
                  ? ` — ${(qualityScore.issues as string[]).slice(0, 2).join("; ")}`
                  : ""}
              </div>
            )}
            <div className="tabs">
              {(
                [
                  ["missing", "Missing"],
                  ["outliers", "Outliers"],
                  ["duplicates", "Duplicates"],
                  ["encoding", "Encoding"],
                  ["scaling", "Scaling"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  className={`tab ${cleanTab === id ? "active" : ""}`}
                  onClick={() => {
                    setCleanTab(id);
                    if (id === "encoding") loadEncoding(false);
                    if (id === "scaling") loadScaling(false);
                    if (id === "outliers") runOutliers("detect");
                  }}
                >
                  {label}
                </button>
              ))}
            </div>

            {cleanTab === "missing" && (
              <>
                <div className="btn-row" style={{ marginTop: 0 }}>
                  <button className="primary" disabled={busy} onClick={() => applyClean("auto")}>
                    Auto-clean (safe)
                  </button>
                  <button className="secondary" disabled={busy} onClick={() => applyClean("all")}>
                    Apply all recommendations
                  </button>
                </div>
                <div style={{ marginTop: "1rem" }}>
                  {recs.filter((r) => r.category === "missing_values" || r.category === "target").length ===
                  0 ? (
                    <div className="success">No missing-value issues detected.</div>
                  ) : (
                    recs
                      .filter((r) => r.category === "missing_values" || r.category === "target")
                      .map((r, i) => (
                        <div className="rec" key={i}>
                          <h4>
                            <span
                              className={`badge ${
                                r.severity === "high"
                                  ? "danger"
                                  : r.severity === "medium"
                                    ? "warn"
                                    : "ok"
                              }`}
                            >
                              {String(r.severity)}
                            </span>{" "}
                            {String(r.issue)}
                          </h4>
                          <p>
                            {String(r.recommendation)}
                            {r.strategy ? ` · strategy=${String(r.strategy)}` : ""}
                          </p>
                        </div>
                      ))
                  )}
                </div>
              </>
            )}

            {cleanTab === "outliers" && (
              <>
                <div className="field">
                  <label>Detection method</label>
                  <select value={outlierMethod} onChange={(e) => setOutlierMethod(e.target.value)}>
                    <option value="iqr">IQR (1.5×)</option>
                    <option value="zscore">Z-Score</option>
                    <option value="isolation_forest">Isolation Forest</option>
                  </select>
                </div>
                <div className="btn-row">
                  <button className="secondary" disabled={busy} onClick={() => runOutliers("detect")}>
                    Detect
                  </button>
                  <button className="primary" disabled={busy} onClick={() => runOutliers("cap")}>
                    Cap outliers
                  </button>
                  <button className="secondary" disabled={busy} onClick={() => runOutliers("drop_rows")}>
                    Drop outlier rows
                  </button>
                </div>
                {outlierReport && (
                  <pre className="mono" style={{ fontSize: "0.8rem", whiteSpace: "pre-wrap" }}>
                    {JSON.stringify(outlierReport, null, 2)}
                  </pre>
                )}
              </>
            )}

            {cleanTab === "duplicates" && (
              <>
                <div className="grid-2">
                  <div className="metric">
                    <div className="label">Duplicate rows</div>
                    <div className="value">
                      {String(profile?.duplicate_rows ?? session.duplicates ?? 0)}
                    </div>
                  </div>
                  <div className="metric">
                    <div className="label">Duplicate %</div>
                    <div className="value">{String(profile?.duplicate_pct ?? 0)}</div>
                  </div>
                </div>
                <div className="btn-row">
                  <button className="primary" disabled={busy} onClick={() => applyClean("auto")}>
                    Drop duplicates (via auto-clean)
                  </button>
                </div>
                {recs
                  .filter((r) => r.category === "duplicates")
                  .map((r, i) => (
                    <div className="rec" key={i}>
                      <h4>{String(r.issue)}</h4>
                      <p>{String(r.recommendation)}</p>
                    </div>
                  ))}
              </>
            )}

            {cleanTab === "encoding" && (
              <>
                <div className="btn-row" style={{ marginTop: 0 }}>
                  <button className="secondary" disabled={busy} onClick={() => loadEncoding(false)}>
                    Refresh recommendations
                  </button>
                  <button className="primary" disabled={busy} onClick={() => loadEncoding(true)}>
                    Apply encoding
                  </button>
                </div>
                {encodingRecs && (
                  <PreviewTable
                    rows={
                      ((encodingRecs.recommendations as Record<string, unknown>[]) || []).map((r) => ({
                        column: r.column,
                        method: r.method,
                        n_unique: r.n_unique,
                        reason: r.reason,
                      }))
                    }
                  />
                )}
              </>
            )}

            {cleanTab === "scaling" && (
              <>
                <div className="btn-row" style={{ marginTop: 0 }}>
                  <button className="secondary" disabled={busy} onClick={() => loadScaling(false)}>
                    Refresh recommendations
                  </button>
                  <button className="primary" disabled={busy} onClick={() => loadScaling(true)}>
                    Apply scaling
                  </button>
                </div>
                {scalingRecs && (
                  <PreviewTable
                    rows={
                      ((scalingRecs.recommendations as Record<string, unknown>[]) || []).map((r) => ({
                        column: r.column,
                        method: r.method,
                        skew: r.skew,
                        reason: r.reason,
                      }))
                    }
                  />
                )}
              </>
            )}

            <PreviewTable rows={session.preview} />
          </div>
        )}

        {step === 4 && session && viz && (
          <div className="panel">
            <div className="tabs">
              {["target", "missing", "categorical", "corr"].map((t) => (
                <button
                  key={t}
                  className={`tab ${vizTab === t ? "active" : ""}`}
                  onClick={() => setVizTab(t)}
                >
                  {t}
                </button>
              ))}
            </div>
            {vizTab === "target" && (
              <div className="chart-box">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={viz.target_distribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="class" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#0d9488" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            {vizTab === "missing" && (
              <div className="chart-box">
                {viz.missing.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={viz.missing} layout="vertical" margin={{ left: 80 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="column" width={80} />
                      <Tooltip />
                      <Bar dataKey="missing_count" fill="#ea580c" radius={[0, 6, 6, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="muted" style={{ padding: "1rem" }}>
                    No missing values
                  </p>
                )}
              </div>
            )}
            {vizTab === "categorical" && (
              <>
                <div className="field">
                  <label>Categorical column</label>
                  <select value={catCol} onChange={(e) => setCatCol(e.target.value)}>
                    {viz.categorical_columns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </div>
                <div className="chart-box">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={catData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="category" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" fill="#0f172a" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
            {vizTab === "corr" && (
              <div className="chart-box" style={{ height: 360 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={viz.feature_correlations.slice(0, 15)} layout="vertical" margin={{ left: 100 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[-1, 1]} />
                    <YAxis type="category" dataKey="feature" width={100} />
                    <Tooltip />
                    <Bar dataKey="correlation" fill="#0d9488" radius={[0, 6, 6, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {step === 5 && session && (
          <div className="panel">
            <div className="grid-3">
              <div className="field">
                <label>Columns to bin</label>
                <div className="checks">
                  {(viz?.numeric_columns || []).map((c) => (
                    <label key={c}>
                      <input
                        type="checkbox"
                        checked={binCols.includes(c)}
                        onChange={(e) =>
                          setBinCols((prev) =>
                            e.target.checked ? [...prev, c] : prev.filter((x) => x !== c)
                          )
                        }
                      />
                      {c}
                    </label>
                  ))}
                </div>
              </div>
              <div className="field">
                <label>Method</label>
                <select value={binMethod} onChange={(e) => setBinMethod(e.target.value)}>
                  <option value="equal_width">equal_width</option>
                  <option value="equal_frequency">equal_frequency</option>
                </select>
              </div>
              <div className="field">
                <label>Bins: {nBins}</label>
                <input
                  type="range"
                  min={3}
                  max={10}
                  value={nBins}
                  onChange={(e) => setNBins(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="btn-row">
              <button className="primary" disabled={busy || !binCols.length} onClick={applyBinning}>
                Apply binning
              </button>
            </div>
            {woe &&
              Object.entries(woe).map(([col, rows]) => (
                <div key={col} style={{ marginTop: "1rem" }}>
                  <h4>WoE / IV — {col}</h4>
                  <PreviewTable rows={rows} />
                </div>
              ))}
          </div>
        )}

        {step === 6 && session && (
          <div className="panel">
            <div className="grid-3">
              <div className="field">
                <label>Method</label>
                <select value={fsMethod} onChange={(e) => setFsMethod(e.target.value)}>
                  {methods.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Top K: {topK}</label>
                <input
                  type="range"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label>Corr threshold: {threshold}</label>
                <input
                  type="range"
                  min={0.01}
                  max={0.9}
                  step={0.01}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="btn-row">
              <button className="primary" disabled={busy} onClick={runFeatures}>
                Run feature selection
              </button>
              <button className="secondary" disabled={busy} onClick={runRf}>
                RF importance
              </button>
              <button className="secondary" disabled={busy} onClick={runFeatureEng}>
                Feature engineering
              </button>
              <button className="secondary" disabled={busy} onClick={() => runReduce(false)}>
                Variable reduction
              </button>
              <button className="secondary" disabled={busy} onClick={() => runReduce(true)}>
                Drop high-corr pairs
              </button>
            </div>
            {feConfig && (
              <div className="success" style={{ marginTop: "0.75rem" }}>
                Engineered {String(feConfig.n_created)} features
                {Array.isArray(feConfig.created)
                  ? `: ${(feConfig.created as string[]).slice(0, 6).join(", ")}`
                  : ""}
              </div>
            )}
            {reduction && (
              <div className="rec" style={{ marginTop: "0.75rem" }}>
                <h4>Variable reduction</h4>
                <p>
                  Drop suggestions: {String(reduction.n_drop_suggestions)} · PCA components for 95%
                  var:{" "}
                  {String(
                    (reduction.pca as Record<string, unknown> | undefined)
                      ?.components_for_target_variance ?? "—"
                  )}
                </p>
                {!!(reduction.drop_suggestions as Record<string, unknown>[] | undefined)?.length && (
                  <PreviewTable
                    rows={(reduction.drop_suggestions as Record<string, unknown>[]).slice(0, 12)}
                  />
                )}
              </div>
            )}
            {!!fsResult.length && scoreKey && (
              <div className="chart-box" style={{ marginTop: "1rem", height: 360 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={[...fsResult].reverse().slice(-15)}
                    layout="vertical"
                    margin={{ left: 110 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="feature" width={110} />
                    <Tooltip />
                    <Bar dataKey={scoreKey} fill="#0d9488" radius={[0, 6, 6, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            {!!fsResult.length && <PreviewTable rows={fsResult} />}
            {!!rfResult.length && (
              <>
                <h4>Random Forest importance</h4>
                <PreviewTable rows={rfResult.slice(0, 20)} />
              </>
            )}
            {!!session.selected_features?.length && (
              <p className="success" style={{ marginTop: "1rem" }}>
                Selected: {session.selected_features.join(", ")}
              </p>
            )}
          </div>
        )}

        {step === 7 && session && (
          <div className="panel">
            <p className="panel-title">Data quality</p>
            <div className="btn-row" style={{ marginTop: 0 }}>
              <button className="secondary" disabled={busy} onClick={runDataValidation}>
                Run data validation
              </button>
            </div>
            {dataReport && (
              <div style={{ margin: "1rem 0" }}>
                <div className="grid-3">
                  <div className="metric">
                    <div className="label">Passed</div>
                    <div className="value">{String(dataReport.passed)}</div>
                  </div>
                  <div className="metric">
                    <div className="label">Warnings</div>
                    <div className="value">{String(dataReport.warnings)}</div>
                  </div>
                  <div className="metric">
                    <div className="label">Failed</div>
                    <div className="value">{String(dataReport.failed)}</div>
                  </div>
                </div>
                {(dataReport.checks as Record<string, unknown>[] | undefined)?.map((c, i) => (
                  <div className="rec" key={i}>
                    <h4>
                      {String(c.status)} — {String(c.name)}
                    </h4>
                    <p>{String(c.message)}</p>
                  </div>
                ))}
              </div>
            )}

            <p className="panel-title" style={{ marginTop: "1.25rem" }}>
              Class balancing / sampling
            </p>
            <div className="btn-row" style={{ marginTop: 0, marginBottom: "0.5rem" }}>
              <button
                className="secondary"
                type="button"
                onClick={() => setSelectedBalances(balanceOptions.map((b) => b.id))}
              >
                Select all balancing
              </button>
              <button className="secondary" type="button" onClick={() => setSelectedBalances(["none"])}>
                None only
              </button>
            </div>
            <div className="checks">
              {balanceOptions.map((b) => (
                <label key={b.id} title={b.description}>
                  <input
                    type="checkbox"
                    checked={selectedBalances.includes(b.id)}
                    onChange={(e) =>
                      setSelectedBalances((prev) =>
                        e.target.checked ? [...prev, b.id] : prev.filter((x) => x !== b.id)
                      )
                    }
                  />
                  {b.label}
                </label>
              ))}
            </div>

            <p className="panel-title" style={{ marginTop: "1.25rem" }}>
              Models
            </p>
            <div className="btn-row" style={{ marginTop: 0, marginBottom: "0.5rem" }}>
              <button
                className="secondary"
                type="button"
                onClick={() => setSelectedModels(modelOptions.map((m) => m.id))}
              >
                Select all models
              </button>
              <button
                className="secondary"
                type="button"
                onClick={() => setSelectedModels(["logistic_regression", "random_forest"])}
              >
                Logistic + RF
              </button>
            </div>
            <div className="checks">
              {modelOptions.map((m) => (
                <label key={m.id}>
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(m.id)}
                    onChange={(e) =>
                      setSelectedModels((prev) =>
                        e.target.checked ? [...prev, m.id] : prev.filter((x) => x !== m.id)
                      )
                    }
                  />
                  {m.label}
                </label>
              ))}
            </div>

            <div className="grid-2" style={{ marginTop: "1rem" }}>
              <div className="field">
                <label>Test size: {testSize}</label>
                <input
                  type="range"
                  min={0.1}
                  max={0.4}
                  step={0.05}
                  value={testSize}
                  onChange={(e) => setTestSize(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label>CV folds: {cvFolds}</label>
                <input
                  type="range"
                  min={2}
                  max={10}
                  value={cvFolds}
                  onChange={(e) => setCvFolds(Number(e.target.value))}
                />
              </div>
            </div>

            <div className="field" style={{ marginTop: "0.5rem" }}>
              <label>Best-algorithm metric</label>
              <select value={selectionMetric} onChange={(e) => setSelectionMetric(e.target.value)}>
                <option value="auc_roc">AUC-ROC (recommended)</option>
                <option value="f1">F1-Score</option>
                <option value="accuracy">Accuracy</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
                <option value="cv_auc_mean">CV AUC mean</option>
              </select>
            </div>

            <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.75rem" }}>
              <input
                type="checkbox"
                checked={runAllCombos}
                onChange={(e) => setRunAllCombos(e.target.checked)}
              />
              Run all combinations (every balancing method × every model)
            </label>

            <div className="btn-row">
              <button className="primary" disabled={busy} onClick={runModelValidation}>
                Train, compare & select best
              </button>
              <span className="muted">
                {selectedBalances.length} balancing × {selectedModels.length} models ={" "}
                {selectedBalances.length * selectedModels.length} runs
              </span>
            </div>

            {selectedAlgorithm && (
              <div className="success" style={{ marginTop: "1rem" }}>
                Selected algorithm:{" "}
                <strong>
                  {String(selectedAlgorithm.model_label)} + {String(selectedAlgorithm.balance_label)}
                </strong>
                {" · "}
                metric={String(selectedAlgorithm.selection_metric)} · by={String(selectedAlgorithm.selected_by)}
                {selectedAlgorithm.metrics ? (
                  <>
                    {" · "}AUC {String((selectedAlgorithm.metrics as Record<string, unknown>).auc_roc)} · F1{" "}
                    {String((selectedAlgorithm.metrics as Record<string, unknown>).f1)}
                  </>
                ) : null}
              </div>
            )}

            <div className="btn-row">
              <button className="secondary" disabled={busy || !selectedAlgorithm} onClick={runTune}>
                Tune hyperparameters
              </button>
              <button className="secondary" disabled={busy || !selectedAlgorithm} onClick={runExplain}>
                Explain model
              </button>
            </div>
            {tuningResult && (
              <div className="rec">
                <h4>Hyperparameter tuning — {String(tuningResult.model_label)}</h4>
                <p>
                  Status {String(tuningResult.status)} · best score {String(tuningResult.best_score)} ·
                  metric {String(tuningResult.selection_metric)}
                </p>
                <pre className="mono" style={{ fontSize: "0.78rem", whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(tuningResult.best_params || {}, null, 2)}
                </pre>
              </div>
            )}
            {explanation && (
              <div style={{ marginTop: "0.75rem" }}>
                <p className="panel-title">
                  Model explanation ({String(explanation.method)})
                </p>
                <div className="chart-box" style={{ height: 280 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[...((explanation.importances as Record<string, unknown>[]) || [])]
                        .slice(0, 12)
                        .reverse()}
                      layout="vertical"
                      margin={{ left: 110 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" width={110} />
                      <Tooltip />
                      <Bar dataKey="importance" fill="#0d9488" radius={[0, 6, 6, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {modelResult && (
              <div style={{ marginTop: "1.25rem" }}>
                <div className="btn-row" style={{ marginTop: 0 }}>
                  <button className="secondary" disabled={busy} onClick={pickBestAlgorithm}>
                    Re-select best by {selectionMetric}
                  </button>
                </div>

                <p className="panel-title">Leaderboard — click Select to choose an algorithm</p>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Model</th>
                        <th>Balancing</th>
                        <th>AUC</th>
                        <th>F1</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>CV AUC</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {((modelResult.leaderboard as Record<string, unknown>[]) || []).map((r, i) => {
                        const m = (r.metrics as Record<string, unknown>) || {};
                        const isSelected =
                          selectedAlgorithm &&
                          selectedAlgorithm.model_id === r.model_id &&
                          selectedAlgorithm.balance_method === r.balance_method;
                        return (
                          <tr key={`${r.model_id}-${r.balance_method}-${i}`} style={isSelected ? { background: "#ecfdf5" } : undefined}>
                            <td>{i + 1}</td>
                            <td>{String(r.model_label)}</td>
                            <td>{String(r.balance_label)}</td>
                            <td>{String(m.auc_roc ?? "")}</td>
                            <td>{String(m.f1 ?? "")}</td>
                            <td>{String(m.accuracy ?? "")}</td>
                            <td>{String(m.precision ?? "")}</td>
                            <td>{String(m.recall ?? "")}</td>
                            <td>{String(m.cv_auc_mean ?? "")}</td>
                            <td>
                              <button
                                className={isSelected ? "primary" : "secondary"}
                                style={{ padding: "0.35rem 0.7rem", fontSize: "0.78rem" }}
                                disabled={busy}
                                onClick={() => pickAlgorithm(String(r.model_id), String(r.balance_method))}
                              >
                                {isSelected ? "Selected" : "Select"}
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <p className="panel-title" style={{ marginTop: "1rem" }}>
                  Split info
                </p>
                <pre className="mono" style={{ fontSize: "0.8rem", whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(modelResult.split, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {step === 8 && session && (
          <div className="panel">
            <div className="grid-4">
              <div className="metric">
                <div className="label">Domain</div>
                <div className="value" style={{ fontSize: "1rem" }}>
                  {session.domain}
                </div>
              </div>
              <div className="metric">
                <div className="label">Target</div>
                <div className="value" style={{ fontSize: "1rem" }}>
                  {session.target}
                </div>
              </div>
              <div className="metric">
                <div className="label">Features</div>
                <div className="value">{session.selected_features?.length || session.columns.length - 1}</div>
              </div>
              <div className="metric">
                <div className="label">Quality</div>
                <div className="value" style={{ fontSize: "1rem" }}>
                  {qualityScore ? `${String(qualityScore.score)} (${String(qualityScore.grade)})` : "—"}
                </div>
              </div>
            </div>
            {selectedAlgorithm ? (
              <div className="success" style={{ marginTop: "1rem" }}>
                Export will use algorithm:{" "}
                <strong>
                  {String(selectedAlgorithm.model_label)} + {String(selectedAlgorithm.balance_label)}
                </strong>
              </div>
            ) : (
              <div className="error" style={{ marginTop: "1rem" }}>
                No algorithm selected yet — run Step 7 to pick the best model.
              </div>
            )}

            {insights && (
              <div style={{ marginTop: "1rem" }}>
                <p className="panel-title">Business insights</p>
                <div className="rec">
                  <h4>Summary</h4>
                  {(insights.summary_bullets as string[] | undefined)?.map((b, i) => (
                    <p key={i}>{b}</p>
                  ))}
                </div>
                {!!(insights.risks as string[] | undefined)?.length && (
                  <div className="rec">
                    <h4>Risks</h4>
                    {(insights.risks as string[]).map((b, i) => (
                      <p key={i}>{b}</p>
                    ))}
                  </div>
                )}
                {!!(insights.opportunities as string[] | undefined)?.length && (
                  <div className="rec">
                    <h4>Opportunities</h4>
                    {(insights.opportunities as string[]).map((b, i) => (
                      <p key={i}>{b}</p>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="btn-row">
              <a className="primary" href={api.exportUrl(session.session_id)} style={{ textDecoration: "none" }}>
                Download ZIP for Predictions Studio
              </a>
              <a
                className="secondary"
                href={api.reportUrl(session.session_id)}
                style={{ textDecoration: "none" }}
              >
                Download executive report (HTML)
              </a>
              <button className="secondary" onClick={loadManifest}>
                Refresh manifest
              </button>
              <button className="secondary" onClick={loadInsights}>
                Refresh insights
              </button>
            </div>
            {manifest && (
              <pre
                className="mono"
                style={{
                  marginTop: "1rem",
                  background: "#fff",
                  border: "1px solid var(--line)",
                  borderRadius: 12,
                  padding: "1rem",
                  overflow: "auto",
                  maxHeight: 420,
                  fontSize: "0.8rem",
                }}
              >
                {JSON.stringify(manifest, null, 2)}
              </pre>
            )}
          </div>
        )}

        {!session && step > 1 && (
          <div className="panel">
            <div className="empty-state">
              <strong>Load data first</strong>
              Go back to Upload & Domain to start the pipeline.
            </div>
            <div className="btn-row" style={{ justifyContent: "center" }}>
              <button className="primary" onClick={() => setStep(1)}>
                Go to Upload
              </button>
            </div>
          </div>
        )}

        <div className="nav-bar">
          <button className="secondary" disabled={step <= 1} onClick={() => setStep((s) => s - 1)}>
            ← Previous
          </button>
          <button className="primary" disabled={step >= 8} onClick={() => setStep((s) => s + 1)}>
            Next →
          </button>
        </div>
      </main>
    </div>
  );
}
