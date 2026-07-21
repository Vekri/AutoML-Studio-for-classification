import { Link } from "react-router-dom";

const FEATURES = [
  {
    title: "Data in. Insights out.",
    text: "Upload CSV or Excel. We automatically detect schemas, quality issues, and target columns.",
  },
  {
    title: "Automatic preparation",
    text: "Clean, encode, scale, engineer, and select features with transparent, reusable pipelines.",
  },
  {
    title: "Models that compete",
    text: "Train and compare logistic regression, random forest, gradient boosting, and more in one view.",
  },
  {
    title: "Explain every prediction",
    text: "Use SHAP, permutation importance, and plain-language business insights to build trust.",
  },
  {
    title: "Deploy in one click",
    text: "Export a production-ready bundle with manifest, model artifacts, and executive HTML report.",
  },
  {
    title: "Monitor what matters",
    text: "Track validation metrics, confusion matrices, and data quality scores from one workspace.",
  },
];

const STEPS = [
  { n: "01", title: "Connect your data", text: "Upload a dataset and choose the outcome you want to predict." },
  { n: "02", title: "Run AutoML", text: "IntelliPredict prepares data, tests models, and finds the strongest result." },
  { n: "03", title: "Explain and deploy", text: "Understand the drivers, export reports, and ship your winning model." },
];

const MODEL_BARS = [
  { name: "XGBoost / Gradient Boosting", pct: 95, color: "violet" },
  { name: "Random Forest", pct: 88, color: "cyan" },
  { name: "Logistic Regression", pct: 67, color: "indigo" },
];

const DRIVERS = [
  { label: "Credit utilization", pct: "34%" },
  { label: "Account tenure", pct: "26%" },
  { label: "Payment history", pct: "21%" },
  { label: "Monthly income", pct: "12%" },
];

export default function LandingPage() {
  return (
    <div className="landing">
      <header className="landing-header">
        <Link to="/" className="landing-brand">
          <span className="landing-logo">IP</span>
          <span>
            IntelliPredict<span className="accent-cyan"> AI</span>
          </span>
        </Link>
        <nav className="landing-nav">
          <a href="#platform">Platform</a>
          <a href="#how-it-works">How it works</a>
          <a href="#security">Security</a>
        </nav>
        <div className="landing-header-actions">
          <Link to="/studio" className="landing-link-btn">
            Sign in
          </Link>
          <Link to="/studio" className="landing-cta-btn">
            Start building →
          </Link>
        </div>
      </header>

      <main>
        <section className="landing-hero">
          <div className="landing-hero-glow landing-hero-glow-a" />
          <div className="landing-hero-glow landing-hero-glow-b" />
          <div className="landing-hero-inner">
            <div className="landing-pill">The complete no-code AutoML workspace</div>
            <h1>
              From raw data to a
              <span className="landing-gradient-text">production-ready model.</span>
            </h1>
            <p className="landing-lead">
              Build, explain, and deploy powerful binary classification models in hours—not weeks.
              No notebooks, infrastructure, or ML team required.
            </p>
            <div className="landing-hero-actions">
              <Link to="/studio" className="landing-primary-btn">
                Build your first model →
              </Link>
              <Link to="/studio" className="landing-secondary-btn">
                Explore the platform
              </Link>
            </div>
            <div className="landing-trust">
              <span>✓ Free to start</span>
              <span>✓ No credit card</span>
              <span>✓ Deploy in minutes</span>
            </div>

            <div className="landing-mockup-wrap">
              <div className="landing-mockup-glow" />
              <div className="landing-mockup">
                <div className="landing-mockup-bar">
                  <span className="dot red" />
                  <span className="dot amber" />
                  <span className="dot green" />
                  <span className="landing-mockup-url">app.intellipredict.ai</span>
                </div>
                <div className="landing-mockup-body">
                  <aside className="landing-mockup-side">
                    <div className="landing-mockup-brand">IntelliPredict</div>
                    <div className="landing-mockup-nav muted">Overview</div>
                    <div className="landing-mockup-nav muted">Data quality</div>
                    <div className="landing-mockup-nav active">Models</div>
                    <div className="landing-mockup-nav muted">Explainability</div>
                    <div className="landing-mockup-nav muted">Deploy</div>
                  </aside>
                  <div className="landing-mockup-main">
                    <div className="landing-mockup-head">
                      <div>
                        <p className="kicker">Model comparison</p>
                        <h3>Find your strongest model</h3>
                        <p className="sub">12 models trained · completed in 2m 18s</p>
                      </div>
                      <span className="status-pill">All systems ready</span>
                    </div>
                    <div className="landing-metrics">
                      <div><p>Best ROC AUC</p><strong>0.947</strong><span className="up">+3.2%</span></div>
                      <div><p>F1 score</p><strong>0.892</strong><span className="up">+1.8%</span></div>
                      <div><p>Accuracy</p><strong>91.4%</strong><span className="up">+2.6%</span></div>
                    </div>
                    <div className="landing-mockup-grid">
                      <div className="landing-chart-card">
                        <div className="card-title">Performance by model <span>ROC AUC</span></div>
                        {MODEL_BARS.map((m) => (
                          <div key={m.name} className="bar-row">
                            <div className="bar-label"><span>{m.name}</span></div>
                            <div className="bar-track"><div className={`bar-fill ${m.color}`} style={{ width: `${m.pct}%` }} /></div>
                          </div>
                        ))}
                      </div>
                      <div className="landing-drivers-card">
                        <div className="card-title">Top prediction drivers</div>
                        {DRIVERS.map((d, i) => (
                          <div key={d.label} className="driver-row">
                            <span className="driver-num">0{i + 1}</span>
                            <span className="driver-label">{d.label}</span>
                            <span className="driver-pct">{d.pct}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="landing-stats">
          <div><strong>15+</strong><span>model algorithms</span></div>
          <div><strong>10K</strong><span>rows per batch</span></div>
          <div><strong>&lt;100ms</strong><span>prediction latency</span></div>
          <div><strong>0 code</strong><span>required to launch</span></div>
        </section>

        <section id="platform" className="landing-section">
          <p className="section-kicker violet">One connected platform</p>
          <h2>Everything you need to turn data into decisions.</h2>
          <p className="section-lead">
            A guided machine-learning workflow that remains powerful enough for experts and clear enough for everyone else.
          </p>
          <div className="landing-features">
            {FEATURES.map((f) => (
              <article key={f.title}>
                <h3>{f.title}</h3>
                <p>{f.text}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="how-it-works" className="landing-section landing-steps-wrap">
          <div className="landing-steps-panel">
            <div className="landing-steps-intro">
              <p className="section-kicker cyan">Simple by design</p>
              <h2>A clear path from question to production.</h2>
              <p>IntelliPredict handles the complexity behind the scenes while keeping every important decision visible.</p>
              <Link to="/studio" className="landing-inline-link">Start with your data →</Link>
            </div>
            <div className="landing-steps">
              {STEPS.map((s) => (
                <div key={s.n} className="landing-step">
                  <span>{s.n}</span>
                  <div>
                    <h3>{s.title}</h3>
                    <p>{s.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="security" className="landing-section landing-security">
          <div>
            <h2>Built for real business data.</h2>
            <p>Session-isolated pipelines, exportable artifacts, and validation metrics built into every workflow.</p>
          </div>
          <div className="landing-security-grid">
            <div><h3>Secure sessions</h3><p>Each upload runs in an isolated server session</p></div>
            <div><h3>Private artifacts</h3><p>Your datasets and models stay in your export bundle</p></div>
            <div><h3>Validation metrics</h3><p>ROC AUC, F1, confusion matrix on held-out data</p></div>
            <div><h3>Reproducible pipelines</h3><p>Every transformation saved in the manifest</p></div>
          </div>
        </section>

        <section className="landing-final-cta">
          <h2>Your next model starts here.</h2>
          <p>Upload your data today and turn a business question into a deployed prediction bundle.</p>
          <Link to="/studio" className="landing-primary-btn light">Get started for free →</Link>
        </section>
      </main>

      <footer className="landing-footer">
        <div className="landing-brand small">IntelliPredict AI</div>
        <p>© 2026 IntelliPredict AI. Built to make machine learning accessible.</p>
        <div className="landing-footer-links">
          <a href="#platform">Platform</a>
          <Link to="/studio">Studio</Link>
        </div>
      </footer>
    </div>
  );
}
