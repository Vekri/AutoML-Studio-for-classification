export default function SettingsPage() {
  return (
    <div className="dash-page">
      <header className="dash-page-head">
        <div>
          <h1>Settings</h1>
          <p>Workspace preferences for IntelliPredict AI.</p>
        </div>
      </header>
      <section className="dash-card">
        <h2>General</h2>
        <p className="muted">
          Projects are stored in your browser local storage. Connect a database in a future release for
          team collaboration.
        </p>
        <ul className="dash-settings-list">
          <li>Default domain: Banking</li>
          <li>Validation metric: ROC AUC</li>
          <li>Export format: ZIP bundle + HTML report</li>
        </ul>
      </section>
    </div>
  );
}
