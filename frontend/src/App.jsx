import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "OTHER"];
const LOAN_INTENT = [
  "EDUCATION",
  "MEDICAL",
  "VENTURE",
  "PERSONAL",
  "HOMEIMPROVEMENT",
  "DEBTCONSOLIDATION",
];

// person_gender and person_education are deliberately absent: the training
// pipeline (data_ingestion.py) drops both columns before the model sees them,
// so asking for them would only add dead fields to the form.
const INITIAL_FORM = {
  person_age: 25,
  person_emp_exp: 0,
  credit_score: 650,
  cb_person_cred_hist_length: 3,
  person_income: 50000,
  loan_amnt: 10000,
  loan_int_rate: 10.0,
  person_home_ownership: "RENT",
  loan_intent: "EDUCATION",
  previous_loan_defaults_on_file: "Yes",
};

function NumberField({ label, name, value, onChange, ...rest }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        name={name}
        value={value}
        onChange={onChange}
        {...rest}
      />
    </label>
  );
}

function SelectField({ label, name, value, options, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select name={name} value={value} onChange={onChange}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function ContributionRow({ contribution, tone }) {
  return (
    <li className="contribution">
      <div className="contribution-labels">
        <span className="contribution-label">{contribution.label}</span>
        <span className="contribution-value">{String(contribution.value)}</span>
      </div>
      <div className="bar">
        <div
          className={`bar-fill ${tone}`}
          style={{ width: `${Math.max(contribution.weight * 100, 2)}%` }}
        />
      </div>
    </li>
  );
}

function Explanation({ explanation }) {
  if (!explanation) return null;

  const { supported, underwhelmed } = explanation;

  return (
    <div className="explanation">
      <h3>Why this decision?</h3>
      <div className="explanation-columns">
        <div>
          <h4 className="supported-heading">Supported approval</h4>
          {supported.length === 0 ? (
            <p className="muted">Nothing pushed this toward approval.</p>
          ) : (
            <ul>
              {supported.map((c) => (
                <ContributionRow key={c.feature} contribution={c} tone="positive" />
              ))}
            </ul>
          )}
        </div>
        <div>
          <h4 className="underwhelmed-heading">Worked against it</h4>
          {underwhelmed.length === 0 ? (
            <p className="muted">Nothing pushed this away from approval.</p>
          ) : (
            <ul>
              {underwhelmed.map((c) => (
                <ContributionRow key={c.feature} contribution={c} tone="negative" />
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function RejectionModal({ data, onClose }) {
  if (!data) return null;
  const isPolicy = data.stage === "policy_gate";

  const overlay = {
    position: "fixed",
    inset: 0,
    background: "rgba(15, 23, 42, 0.55)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "20px",
    zIndex: 1000,
  };
  const card = {
    background: "#ffffff",
    color: "#0f172a",
    borderRadius: "16px",
    maxWidth: "440px",
    width: "100%",
    padding: "34px 30px 30px",
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.35)",
    textAlign: "center",
    position: "relative",
    borderTop: "5px solid #dc2626",
  };
  const badge = {
    width: "56px",
    height: "56px",
    borderRadius: "50%",
    background: "#fee2e2",
    color: "#dc2626",
    fontSize: "26px",
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    margin: "0 auto 16px",
  };
  const closeX = {
    position: "absolute",
    top: "12px",
    right: "14px",
    border: "none",
    background: "transparent",
    fontSize: "22px",
    lineHeight: 1,
    cursor: "pointer",
    color: "#94a3b8",
  };
  const okBtn = {
    marginTop: "24px",
    padding: "10px 26px",
    border: "none",
    borderRadius: "8px",
    background: "#dc2626",
    color: "#ffffff",
    fontWeight: 600,
    cursor: "pointer",
  };
  const body = { margin: 0, lineHeight: 1.6, color: "#334155" };

  return (
    <div style={overlay} role="dialog" aria-modal="true" onClick={onClose}>
      <div style={card} onClick={(event) => event.stopPropagation()}>
        <button type="button" style={closeX} aria-label="Close" onClick={onClose}>
          ×
        </button>
        <div style={badge}>✕</div>
        <h2 style={{ margin: "0 0 10px", color: "#dc2626" }}>Application Rejected</h2>
        {isPolicy ? (
          <p style={body}>
            Thank you for your application. Unfortunately, we are unable to approve it
            at this time. In accordance with our lending policy, we cannot extend
            credit to applicants with a previous loan default on file. We appreciate
            your understanding and welcome you to apply again once your credit profile
            has strengthened.
          </p>
        ) : (
          <p style={body}>
            Thank you for your application. Unfortunately, it does not meet our current
            approval criteria. You are welcome to review the contributing factors on
            the decision panel and to reapply in the future.
          </p>
        )}
        <button type="button" style={okBtn} onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [rejectModal, setRejectModal] = useState(null);

  // loan_percent_income is fully determined by loan_amnt / person_income, so it
  // is computed rather than being a free-form input the user can make
  // inconsistent with the two fields it derives from.
  const loanPercentIncome =
    form.person_income > 0
      ? Number((form.loan_amnt / form.person_income).toFixed(4))
      : 0;

  function handleChange(event) {
    const { name, value, type } = event.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "number" ? (value === "" ? "" : Number(value)) : value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const payload = {
      person_age: form.person_age,
      person_income: form.person_income,
      person_emp_exp: form.person_emp_exp,
      person_home_ownership: form.person_home_ownership,
      loan_amnt: form.loan_amnt,
      loan_intent: form.loan_intent,
      loan_int_rate: form.loan_int_rate,
      loan_percent_income: loanPercentIncome,
      cb_person_cred_hist_length: form.cb_person_cred_hist_length,
      credit_score: form.credit_score,
      previous_loan_defaults_on_file: form.previous_loan_defaults_on_file,
    };

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `API error (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
      if (data.prediction === 0) setRejectModal(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? `Could not reach the API at ${API_URL}. Is the FastAPI server running?`
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setForm(INITIAL_FORM);
    setResult(null);
    setError(null);
    setRejectModal(null);
  }

  return (
    <div className="page">
      <header>
        <span className="eyebrow">Credit decisioning</span>
        <h1>Loan Approval Prediction</h1>
        <p>Enter the applicant details to get an instant, explainable decision.</p>
      </header>

      <div className="layout">
      <form onSubmit={handleSubmit} className="panel form-panel">
        <fieldset disabled={loading}>
          <legend>Applicant</legend>
          <div className="grid">
            <NumberField
              label="Age"
              name="person_age"
              value={form.person_age}
              onChange={handleChange}
              min={18}
              max={100}
              required
            />
            <NumberField
              label="Work Experience (years)"
              name="person_emp_exp"
              value={form.person_emp_exp}
              onChange={handleChange}
              min={0}
              max={50}
              required
            />
            <NumberField
              label="Annual Income"
              name="person_income"
              value={form.person_income}
              onChange={handleChange}
              min={0}
              step={1000}
              required
            />
            <NumberField
              label="Credit Score"
              name="credit_score"
              value={form.credit_score}
              onChange={handleChange}
              min={300}
              max={900}
              required
            />
            <NumberField
              label="Credit History Length (years)"
              name="cb_person_cred_hist_length"
              value={form.cb_person_cred_hist_length}
              onChange={handleChange}
              min={0}
              max={50}
              required
            />
            <SelectField
              label="Home Ownership"
              name="person_home_ownership"
              value={form.person_home_ownership}
              options={HOME_OWNERSHIP}
              onChange={handleChange}
            />
          </div>
        </fieldset>

        <fieldset disabled={loading}>
          <legend>Loan</legend>
          <div className="grid">
            <NumberField
              label="Loan Amount"
              name="loan_amnt"
              value={form.loan_amnt}
              onChange={handleChange}
              min={0}
              step={500}
              required
            />
            <NumberField
              label="Interest Rate (%)"
              name="loan_int_rate"
              value={form.loan_int_rate}
              onChange={handleChange}
              min={0}
              step={0.1}
              required
            />
            <SelectField
              label="Loan Intent"
              name="loan_intent"
              value={form.loan_intent}
              options={LOAN_INTENT}
              onChange={handleChange}
            />
            <SelectField
              label="Previous Default"
              name="previous_loan_defaults_on_file"
              value={form.previous_loan_defaults_on_file}
              options={["Yes", "No"]}
              onChange={handleChange}
            />
            <div className="field computed">
              <span>Loan % of Income</span>
              <output>{(loanPercentIncome * 100).toFixed(2)}%</output>
              <small>Computed from loan amount ÷ income</small>
            </div>
          </div>
        </fieldset>

        <div className="actions">
          <button type="submit" disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" /> Predicting…
              </>
            ) : (
              "Predict"
            )}
          </button>
          <button type="button" className="secondary" onClick={handleReset}>
            Reset
          </button>
        </div>
      </form>

      <div className="panel result-panel">
        {error && <div className="alert error">{error}</div>}

        {!result && !error && (
          <div className="empty-state">
            <div className="empty-icon">→</div>
            <h3>No decision yet</h3>
            <p>Fill in the applicant and loan details, then hit Predict.</p>
          </div>
        )}

        {result && (
          <div
            className={`result ${result.prediction === 1 ? "approved" : "rejected"}`}
          >
            <button
              type="button"
              className="result-close"
              aria-label="Dismiss result"
              onClick={() => setResult(null)}
            >
              ×
            </button>
            <div className="result-badge">
              {result.prediction === 1 ? "✓" : "✕"}
            </div>
            <h2>{result.prediction === 1 ? "Approved" : "Rejected"}</h2>
            <p className="probability">
              Approval probability: <strong>{(result.probability * 100).toFixed(2)}%</strong>
            </p>
            <div className="bar">
              <div
                className="bar-fill"
                style={{ width: `${Math.min(result.probability * 100, 100)}%` }}
              />
            </div>
            <Explanation explanation={result.explanation} />
          </div>
        )}
      </div>
      </div>

      <RejectionModal data={rejectModal} onClose={() => setRejectModal(null)} />
    </div>
  );
}
