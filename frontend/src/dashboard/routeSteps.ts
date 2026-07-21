export const DASHBOARD_NAV = [
  { path: "/dashboard/projects", label: "Projects", step: null },
  { path: "/dashboard/upload", label: "Upload", step: 1 },
  { path: "/dashboard/quality", label: "Quality", step: 3 },
  { path: "/dashboard/eda", label: "EDA", step: 4 },
  { path: "/dashboard/engineering", label: "Engineering", step: 5 },
  { path: "/dashboard/selection", label: "Selection", step: 6 },
  { path: "/dashboard/models", label: "Models", step: 7 },
  { path: "/dashboard/explain", label: "Explain", step: 7 },
  { path: "/dashboard/reports", label: "Reports", step: 8 },
  { path: "/dashboard/deploy", label: "Deploy", step: 8 },
  { path: "/dashboard/settings", label: "Settings", step: null },
] as const;

export function stepForDashboardPath(pathname: string): number | null {
  const item = DASHBOARD_NAV.find((n) => pathname === n.path || pathname.startsWith(`${n.path}/`));
  return item?.step ?? null;
}
