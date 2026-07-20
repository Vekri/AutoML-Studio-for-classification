export type DomainInfo = {
  name: string;
  description: string;
  common_targets: string[];
  key_metrics: string[];
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
    request<{ domains: DomainInfo[]; feature_selection_methods: string[] }>("/api/domains"),
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
  validateModels: (sessionId: string, body: { test_size: number; cv_folds: number }) =>
    request<{ result: Record<string, unknown> }>(`/api/session/${sessionId}/validate/models`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  manifest: (sessionId: string) =>
    request<Record<string, unknown>>(`/api/session/${sessionId}/manifest`),
  exportUrl: (sessionId: string) => `/api/session/${sessionId}/export.zip`,
  reset: (sessionId: string) =>
    request<{ ok: boolean }>(`/api/session/${sessionId}`, { method: "DELETE" }),
};
