import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createProject, deleteProject, listProjects, Project } from "./projectStore";
import { PROJECT_TEMPLATES } from "./templates";

export default function ProjectsPage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [name, setName] = useState("");
  const [templateId, setTemplateId] = useState("blank");

  useEffect(() => {
    setProjects(listProjects());
    setLoading(false);
  }, []);

  function refresh() {
    setProjects(listProjects());
  }

  function onCreate(e: FormEvent) {
    e.preventDefault();
    const project = createProject(name, templateId);
    setName("");
    setTemplateId("blank");
    refresh();
    navigate(`/dashboard/upload?project=${project.id}`);
  }

  function onOpen(project: Project) {
    navigate(`/dashboard/upload?project=${project.id}`);
  }

  function onRemove(id: string) {
    deleteProject(id);
    refresh();
  }

  return (
    <div className="dash-page">
      <header className="dash-page-head">
        <div>
          <h1>Projects</h1>
          <p>Create and manage binary classification projects.</p>
        </div>
      </header>

      <section className="dash-card dash-new-project">
        <h2>New Project</h2>
        <form className="dash-form" onSubmit={onCreate}>
          <label>
            Project name
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My classification project"
            />
          </label>
          <label>
            Template
            <select value={templateId} onChange={(e) => setTemplateId(e.target.value)}>
              {PROJECT_TEMPLATES.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
          <button type="submit" className="dash-primary-btn">
            Create Project
          </button>
        </form>
      </section>

      <section className="dash-card">
        <h2>Your projects</h2>
        {loading ? (
          <p className="muted">Loading projects...</p>
        ) : projects.length === 0 ? (
          <p className="muted">No projects yet. Create one above to get started.</p>
        ) : (
          <div className="dash-project-grid">
            {projects.map((p) => {
              const template = PROJECT_TEMPLATES.find((t) => t.id === p.templateId);
              return (
                <article key={p.id} className="dash-project-card">
                  <div>
                    <h3>{p.name}</h3>
                    <p>{template?.label || "Custom"} · {p.domain}</p>
                    <p className="muted small">
                      Updated {new Date(p.updatedAt).toLocaleString()}
                    </p>
                  </div>
                  <div className="dash-project-actions">
                    <button type="button" className="dash-primary-btn" onClick={() => onOpen(p)}>
                      Open
                    </button>
                    <button type="button" className="dash-ghost-btn" onClick={() => onRemove(p.id)}>
                      Delete
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
