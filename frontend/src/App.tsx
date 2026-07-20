import { useEffect, useMemo, useState } from "react";
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

export default function App() {
  const [step, setStep] = useState(1);
  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [methods, setMethods] = useState<string[]>([]);
  const [balanceOptions, setBalanceOptions] = useState<BalanceMethod[]>([]);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [domain, setDomain] = useState("Banking");
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

  // Step 8
  const [manifest, setManifest] = useState<Record<string, unknown> | null>(null);

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
    setTarget(data.suggested_target || data.target || "default");
    setSelectedCols(data.columns.filter((c) => c !== (data.suggested_target || "default") && !c.toLowerCase().includes("id")));
    setStep(2);
  }

  async function onUpload(file: File | null) {
    if (!file) return;
    const data = await run(() => api.upload(file, domain), "CSV uploaded");
    if (!data) return;
    setSession(data);
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
      await loadCleaning();
    }
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
        }),
      "Model suite complete"
    );
    if (data) setModelResult(data.result);
  }

  async function loadManifest() {
    if (!session) return;
    const data = await run(() => api.manifest(session.session_id));
    if (data) setManifest(data);
  }

  useEffect(() => {
    if (step === 8 && session) loadManifest();
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
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">AS</div>
          <div>
            <h1>AutoML Studio</h1>
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

      <main className="main" key={step}>
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
              <p className="panel-title">Dataset overview</p>
              {session ? (
                <div className="split-stack">
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
                      <div className="label">Missing</div>
                      <div className="value">{session.missing_cells ?? 0}</div>
                    </div>
                    <div className="metric">
                      <div className="label">Duplicates</div>
                      <div className="value">{session.duplicates ?? 0}</div>
                    </div>
                  </div>
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
            <div className="btn-row" style={{ marginTop: 0 }}>
              <button className="primary" disabled={busy} onClick={() => applyClean("auto")}>
                Auto-clean (safe)
              </button>
              <button className="secondary" disabled={busy} onClick={() => applyClean("all")}>
                Apply all recommendations
              </button>
            </div>
            <div style={{ marginTop: "1rem" }}>
              {recs.length === 0 ? (
                <div className="success">No major cleaning issues detected.</div>
              ) : (
                recs.map((r, i) => (
                  <div className="rec" key={i}>
                    <h4>
                      <span
                        className={`badge ${
                          r.severity === "high" ? "danger" : r.severity === "medium" ? "warn" : "ok"
                        }`}
                      >
                        {String(r.severity)}
                      </span>{" "}
                      [{String(r.category)}] {String(r.issue)}
                    </h4>
                    <p>{String(r.recommendation)}</p>
                  </div>
                ))
              )}
            </div>
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
            </div>
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
                Train & compare models
              </button>
              <span className="muted">
                {selectedBalances.length} balancing × {selectedModels.length} models ={" "}
                {selectedBalances.length * selectedModels.length} runs
              </span>
            </div>

            {modelResult && (
              <div style={{ marginTop: "1.25rem" }}>
                {modelResult.best && (
                  <div className="success">
                    Best:{" "}
                    <strong>
                      {String((modelResult.best as Record<string, unknown>).model_label)} +{" "}
                      {String((modelResult.best as Record<string, unknown>).balance_label)}
                    </strong>{" "}
                    (AUC{" "}
                    {String(
                      ((modelResult.best as Record<string, unknown>).metrics as Record<string, unknown>)
                        ?.auc_roc ?? "N/A"
                    )}
                    )
                  </div>
                )}

                <p className="panel-title">Leaderboard</p>
                <PreviewTable
                  rows={((modelResult.leaderboard as Record<string, unknown>[]) || []).map((r) => {
                    const m = (r.metrics as Record<string, unknown>) || {};
                    return {
                      rank_model: r.model_label,
                      balancing: r.balance_label,
                      auc_roc: m.auc_roc,
                      f1: m.f1,
                      accuracy: m.accuracy,
                      precision: m.precision,
                      recall: m.recall,
                      cv_auc: m.cv_auc_mean,
                    };
                  })}
                />

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
                <div className="label">Rows</div>
                <div className="value">{session.rows}</div>
              </div>
            </div>
            <div className="btn-row">
              <a className="primary" href={api.exportUrl(session.session_id)} style={{ textDecoration: "none" }}>
                Download ZIP for Predictions Studio
              </a>
              <button className="secondary" onClick={loadManifest}>
                Refresh manifest
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
