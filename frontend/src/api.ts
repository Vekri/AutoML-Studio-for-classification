export type DomainInfo = {
  name: string;
  description: string;
  common_targets: string[];
  key_metrics: string[];
};

export type BalanceMethod = {
  id: string;
  label: string;
  description: string;
};

export type ModelOption = {
  id: string;
  label: string;
  supports_class_weight: boolean;
};

export type OutlierMethod = {
  id: string;
  label: string;
  description: string;
};

export type SessionPayload = {
  session_id: string;
  filename: string;
  domain: string;
  target: string | null;
  rows: number;
  columns: string[];
  preview: Record<string, unknown>[];
  selected_features: string[];
  has_cleaning: boolean;
  has_binning: boolean;
  suggested_target?: string | null;
  column_summary?: Record<string, unknown>[];
  missing_cells?: number;
  duplicates?: number;
  target_info?: Record<string, unknown>;
  cleaning_config?: Record<string, unknown>;
  binning_config?: Record<string, unknown>;
  woe_tables?: Record<string, Record<string, unknown>[]>;
  result?: Record<string, unknown>[];
  profile?: Record<string, unknown>;
  quality_score?: Record<string, unknown> | null;
  selected_algorithm?: Record<string, unknown> | null;
  has_encoding?: boolean;
  has_scaling?: boolean;
  has_feature_eng?: boolean;
  has_explanation?: boolean;
  has_tuning?: boolean;
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  domains: () =>
    request<{
      domains: DomainInfo[];
      feature_selection_methods: string[];
      balance_methods: BalanceMethod[];
      models: ModelOption[];
      outlier_methods?: OutlierMethod[];
    }>("/api/domains"),
  sample: (domain = "Banking") =>
    request<SessionPayload>(`/api/sample?domain=${encodeURIComponent(domain)}`, { method: "POST" }),
  upload: async (file: File, domain: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("domain", domain);
    return request<SessionPayload>("/api/upload", { method: "POST", body: form });
  },
  updateDomain: (sessionId: string, domain: string) =>
    request<SessionPayload>(`/api/session/${sessionId}/domain`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain }),
    }),
  targetInfo: (sessionId: string, target: string) =>
    request<{ info: Record<string, unknown>; distribution: { class: string; count: number }[] }>(
      `/api/session/${sessionId}/target-info?target=${encodeURIComponent(target)}`
    ),
  selectColumns: (sessionId: string, body: { target: string; mode: string; columns: string[] }) =>
    request<SessionPayload>(`/api/session/${sessionId}/columns`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  cleaning: (sessionId: string) =>
    request<{ recommendations: Record<string, unknown>[]; config: Record<string, unknown> }>(
      `/api/session/${sessionId}/cleaning`
    ),
  applyCleaning: (sessionId: string, mode: "auto" | "all") =>
    request<SessionPayload>(`/api/session/${sessionId}/cleaning/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    }),
  profile: (sessionId: string, refresh = false) =>
    request<{ profile: Record<string, unknown> } & SessionPayload>(
      `/api/session/${sessionId}/profile?refresh=${refresh}`
    ),
  qualityScore: (sessionId: string) =>
    request<{ quality_score: Record<string, unknown> } & SessionPayload>(
      `/api/session/${sessionId}/quality-score`
    ),
  outliers: (
    sessionId: string,
    body: { method: string; mode: string; z_threshold?: number; contamination?: number }
  ) =>
    request<{ report: Record<string, unknown> } & SessionPayload>(
      `/api/session/${sessionId}/outliers`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    ),
  encoding: (sessionId: string, apply = false) =>
    request<{ recommendations: Record<string, unknown>; config?: Record<string, unknown> } & SessionPayload>(
      `/api/session/${sessionId}/encoding`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apply }),
      }
    ),
  scaling: (sessionId: string, apply = false, algorithm_hint?: string) =>
    request<{ recommendations: Record<string, unknown>; config?: Record<string, unknown> } & SessionPayload>(
      `/api/session/${sessionId}/scaling`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apply, algorithm_hint }),
      }
    ),
  featureEng: (sessionId: string, body?: { max_interactions?: number; include_datetime?: boolean }) =>
    request<{ config: Record<string, unknown> } & SessionPayload>(
      `/api/session/${sessionId}/feature-eng`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || { max_interactions: 5, include_datetime: true }),
      }
    ),
  reduce: (
    sessionId: string,
    body?: {
      corr_threshold?: number;
      run_pca?: boolean;
      apply_drops?: boolean;
      drop_columns?: string[];
    }
  ) =>
    request<{ reduction: Record<string, unknown> } & SessionPayload>(
      `/api/session/${sessionId}/reduce`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || { corr_threshold: 0.92, run_pca: true, apply_drops: false }),
      }
    ),
  tune: (sessionId: string, body?: { selection_metric?: string; n_iter?: number; cv_folds?: number }) =>
    request<{ tuning: Record<string, unknown>; selected_algorithm?: Record<string, unknown> | null }>(
      `/api/session/${sessionId}/tune`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || { selection_metric: "auc_roc", n_iter: 20, cv_folds: 3 }),
      }
    ),
  explain: (sessionId: string, top_k = 15) =>
    request<{ explanation: Record<string, unknown> }>(`/api/session/${sessionId}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ top_k }),
    }),
  insights: (sessionId: string) =>
    request<{ insights: Record<string, unknown> }>(`/api/session/${sessionId}/insights`),
  reportUrl: (sessionId: string) => `/api/session/${sessionId}/report.html`,
  visualizations: (sessionId: string) =>
    request<{
      target_distribution: { class: string; count: number }[];
      missing: { column: string; missing_count: number }[];
      numeric_columns: string[];
      categorical_columns: string[];
      feature_correlations: { feature: string; correlation: number }[];
    }>(`/api/session/${sessionId}/visualizations`),
  vizNumeric: (sessionId: string, column: string) =>
    request<{ column: string; points: { value: number; target?: string }[] }>(
      `/api/session/${sessionId}/visualizations/numeric/${encodeURIComponent(column)}`
    ),
  vizCategorical: (sessionId: string, column: string) =>
    request<{ column: string; grouped: Record<string, unknown>[] }>(
      `/api/session/${sessionId}/visualizations/categorical/${encodeURIComponent(column)}`
    ),
  binning: (sessionId: string, body: { columns: string[]; method: string; n_bins: number }) =>
    request<SessionPayload>(`/api/session/${sessionId}/binning`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  features: (
    sessionId: string,
    body: { method: string; top_k: number; threshold: number }
  ) =>
    request<SessionPayload>(`/api/session/${sessionId}/features`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  rfImportance: (sessionId: string) =>
    request<{ result: Record<string, unknown>[] }>(
      `/api/session/${sessionId}/features/rf-importance`,
      { method: "POST" }
    ),
  validateData: (sessionId: string) =>
    request<{ report: Record<string, unknown>; features_used: string[] }>(
      `/api/session/${sessionId}/validate/data`,
      { method: "POST" }
    ),
  validateModels: (
    sessionId: string,
    body: {
      test_size: number;
      cv_folds: number;
      models: string[];
      balance_methods: string[];
      run_all_combinations: boolean;
      selection_metric?: string;
      auto_select_best?: boolean;
    }
  ) =>
    request<{ result: Record<string, unknown>; selected_algorithm?: Record<string, unknown> | null }>(
      `/api/session/${sessionId}/validate/models`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    ),
  selectAlgorithm: (
    sessionId: string,
    body: { model_id: string; balance_method: string; selection_metric?: string }
  ) =>
    request<{ selected_algorithm: Record<string, unknown> }>(
      `/api/session/${sessionId}/select-algorithm`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    ),
  autoSelectAlgorithm: (sessionId: string, selection_metric = "auc_roc") =>
    request<{ selected_algorithm: Record<string, unknown> }>(
      `/api/session/${sessionId}/select-algorithm/auto`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selection_metric }),
      }
    ),
  getSelectedAlgorithm: (sessionId: string) =>
    request<{ selected_algorithm: Record<string, unknown> | null }>(
      `/api/session/${sessionId}/selected-algorithm`
    ),
  manifest: (sessionId: string) =>
    request<Record<string, unknown>>(`/api/session/${sessionId}/manifest`),
  exportUrl: (sessionId: string) => `/api/session/${sessionId}/export.zip`,
  reset: (sessionId: string) =>
    request<{ ok: boolean }>(`/api/session/${sessionId}`, { method: "DELETE" }),
};
