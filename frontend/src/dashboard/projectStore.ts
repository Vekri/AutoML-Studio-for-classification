import { ProjectTemplate, templateById } from "./templates";

export type Project = {
  id: string;
  name: string;
  templateId: string;
  domain: string;
  createdAt: string;
  updatedAt: string;
  sessionId?: string;
  status: "draft" | "ready" | "trained";
};

const STORAGE_KEY = "intellipredict_projects";

function readAll(): Project[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Project[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeAll(projects: Project[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(projects));
}

export function listProjects(): Project[] {
  return readAll().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

export function getProject(id: string): Project | undefined {
  return readAll().find((p) => p.id === id);
}

export function createProject(name: string, templateId: string): Project {
  const template: ProjectTemplate = templateById(templateId) || templateById("blank")!;
  const now = new Date().toISOString();
  const project: Project = {
    id: crypto.randomUUID(),
    name: name.trim() || template.label,
    templateId: template.id,
    domain: template.domain,
    createdAt: now,
    updatedAt: now,
    status: "draft",
  };
  const projects = readAll();
  projects.unshift(project);
  writeAll(projects);
  return project;
}

export function updateProject(id: string, patch: Partial<Project>): Project | undefined {
  const projects = readAll();
  const idx = projects.findIndex((p) => p.id === id);
  if (idx < 0) return undefined;
  projects[idx] = { ...projects[idx], ...patch, updatedAt: new Date().toISOString() };
  writeAll(projects);
  return projects[idx];
}

export function deleteProject(id: string) {
  writeAll(readAll().filter((p) => p.id !== id));
}
