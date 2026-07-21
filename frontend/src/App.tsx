import { Navigate, Route, Routes } from "react-router-dom";
import LandingPage from "./LandingPage";
import StudioApp from "./StudioApp";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/studio" element={<StudioApp />} />
      <Route path="/dashboard" element={<Navigate to="/studio" replace />} />
      <Route path="/dashboard/*" element={<Navigate to="/studio" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
