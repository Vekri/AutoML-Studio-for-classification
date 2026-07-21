import { useMemo } from "react";
import { useLocation } from "react-router-dom";
import StudioApp from "../StudioApp";
import { getProject } from "./projectStore";
import { stepForDashboardPath } from "./routeSteps";
import { templateById } from "./templates";

export default function DashboardStudio() {
  const { pathname, search } = useLocation();
  const projectId = new URLSearchParams(search).get("project");
  const project = projectId ? getProject(projectId) : undefined;
  const template = project ? templateById(project.templateId) : undefined;

  const initialStep = useMemo(() => stepForDashboardPath(pathname) ?? 1, [pathname]);

  return (
    <StudioApp
      embedded
      initialStep={initialStep}
      projectId={project?.id}
      projectName={project?.name}
      domain={project?.domain}
      loadSampleOnStart={Boolean(template?.sample)}
    />
  );
}
