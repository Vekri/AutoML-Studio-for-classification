import { Navigate, Route, Routes } from "react-router-dom";
import DashboardLayout from "./dashboard/DashboardLayout";
import DashboardStudio from "./dashboard/DashboardStudio";
import ProjectsPage from "./dashboard/ProjectsPage";
import SettingsPage from "./dashboard/SettingsPage";
import LandingPage from "./LandingPage";
import StudioApp from "./StudioApp";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/studio" element={<StudioApp />} />
      <Route path="/dashboard" element={<DashboardLayout />}>
        <Route index element={<Navigate to="/dashboard/projects" replace />} />
        <Route path="projects" element={<ProjectsPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="upload" element={<DashboardStudio />} />
        <Route path="quality" element={<DashboardStudio />} />
        <Route path="eda" element={<DashboardStudio />} />
        <Route path="engineering" element={<DashboardStudio />} />
        <Route path="selection" element={<DashboardStudio />} />
        <Route path="models" element={<DashboardStudio />} />
        <Route path="explain" element={<DashboardStudio />} />
        <Route path="reports" element={<DashboardStudio />} />
        <Route path="deploy" element={<DashboardStudio />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
