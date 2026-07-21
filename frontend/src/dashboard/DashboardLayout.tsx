import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { DASHBOARD_NAV } from "./routeSteps";
import { getProject } from "./projectStore";

function useActiveProjectName() {
  const { search } = useLocation();
  const id = new URLSearchParams(search).get("project");
  if (!id) return null;
  return getProject(id)?.name ?? null;
}

export default function DashboardLayout() {
  const projectName = useActiveProjectName();
  const location = useLocation();

  return (
    <div className="dash-shell">
      <aside className="dash-sidebar">
        <Link to="/" className="dash-brand">
          <span className="dash-logo">IP</span>
          <span>
            IntelliPredict<span className="accent-cyan"> AI</span>
          </span>
        </Link>
        {projectName && <div className="dash-project-pill">{projectName}</div>}
        <nav className="dash-nav">
          {DASHBOARD_NAV.map((item) => {
            const qs = location.search;
            const to = item.path === "/dashboard/projects" ? item.path : `${item.path}${qs}`;
            return (
              <NavLink
                key={item.path}
                to={to}
                className={({ isActive }) => `dash-nav-link${isActive ? " active" : ""}`}
              >
                {item.label}
              </NavLink>
            );
          })}
        </nav>
        <div className="dash-sidebar-foot">
          <Link to="/" className="dash-nav-link muted">
            ← Marketing site
          </Link>
        </div>
      </aside>
      <main className="dash-main">
        <Outlet />
      </main>
    </div>
  );
}
