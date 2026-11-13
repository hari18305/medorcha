import { useState } from "react";
import ErrorBanner from "../components/ErrorBanner.jsx";
import ReportSummary from "../components/ReportSummary.jsx";
import { ArrowLeftIcon, CheckIcon } from "../components/icons.jsx";
import { submitReview } from "../services/api.js";

const DECISIONS = [
  { value: "accept", label: "Accept", description: "The differential above is correct as presented." },
  { value: "override", label: "Override", description: "Replace the top diagnosis with a different one." },
  { value: "annotate", label: "Annotate", description: "Leave a note without changing the diagnosis." },
];

/**
 * Full HITL review screen: shows the same report data (confidence,
 * conflicts, diagnoses, evidence) via the shared ReportSummary, with the
 * three review actions in its sidebar. Reachable both automatically
 * (App.jsx routes here when report.review_required is true) and manually
 * (the "Review case" button on the Report screen).
 *
 * "Override" has no dedicated backend field for the replacement
 * diagnosis -- ReviewRequest only has {decision, annotation, reviewer}
 * -- so the chosen/typed diagnosis is written into `annotation` as
 * "Override diagnosis: <name>", matching the convention documented in
 * backend/api/schemas.py.
 */
export default function Review({ report, submittedSummary, onBack, onReviewed }) {
  const [decision, setDecision] = useState("accept");
  const [overrideDiagnosis, setOverrideDiagnosis] = useState("");
  const [annotationNote, setAnnotationNote] = useState("");
  const [reviewer, setReviewer] = useState("");
  // Optional 1-5 rating (Phase 11) -- null means "not rated", not 0;
  // never required for submission.
  const [rating, setRating] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [confirmation, setConfirmation] = useState(null);

  const diagnoses = [...(report.diagnoses ?? [])].sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0));

  async function handleSubmit(event) {
    event.preventDefault();
    setError(null);

    if (!reviewer.trim()) {
      setError("Reviewer name is required.");
      return;
    }

    let annotation = null;
    if (decision === "override") {
      if (!overrideDiagnosis.trim()) {
        setError("Pick or type the diagnosis you're overriding to.");
        return;
      }
      annotation = `Override diagnosis: ${overrideDiagnosis.trim()}`;
    } else if (decision === "annotate") {
      if (!annotationNote.trim()) {
        setError("An annotation note is required.");
        return;
      }
      annotation = annotationNote.trim();
    }

    setSubmitting(true);
    try {
      const result = await submitReview(report.case_id, {
        decision,
        annotation,
        reviewer: reviewer.trim(),
        rating,
      });
      setConfirmation(result);
    } catch (err) {
      setError(err); // an ApiError -- ErrorBanner words network / 422 / server failures differently
    } finally {
      setSubmitting(false);
    }
  }

  if (confirmation) {
    return (
      <div className="mx-auto max-w-xl animate-rise">
        <div className="card card-pad text-center">
          <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-emerald-100 text-emerald-700">
            <CheckIcon className="h-6 w-6" strokeWidth="2.25" />
          </span>
          <h2 className="mt-4 text-xl font-semibold tracking-tight text-slate-900">Case closed</h2>
          <p className="mt-2 text-sm text-slate-600">
            {confirmation.reviewer} recorded <span className="font-semibold">{confirmation.decision}</span> for case{" "}
            <span className="font-mono text-xs">{confirmation.case_id}</span>, and the case is now closed.
          </p>
          {confirmation.annotation && (
            <p className="mt-4 rounded-lg bg-slate-50 px-3 py-2.5 text-left text-sm text-slate-700 ring-1 ring-inset ring-slate-200">
              {confirmation.annotation}
            </p>
          )}
          {confirmation.rating != null && (
            <p className="mt-3 text-sm text-slate-600">Review confidence: {confirmation.rating} / 5</p>
          )}
          <button type="button" onClick={onReviewed} className="btn-secondary mt-6 w-full">
            <ArrowLeftIcon className="h-4 w-4" />
            Back to report
          </button>
        </div>
      </div>
    );
  }

  const decisionPanel = (
    <section className="card card-pad" data-testid="decision-panel">
      <h3 className="text-sm font-semibold text-slate-900">Clinician decision</h3>
      <p className="field-hint">Your decision closes the case and is recorded with your name.</p>

      {error && (
        <div className="mt-3">
          <ErrorBanner error={error} />
        </div>
      )}

      <form onSubmit={handleSubmit} className="mt-4 space-y-5">
        <div className="grid gap-2" role="radiogroup" aria-label="Review decision">
          {DECISIONS.map((option) => (
            <label
              key={option.value}
              className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 text-sm transition focus-within:ring-2 focus-within:ring-brand-500/40 ${
                decision === option.value
                  ? "border-brand-500 bg-brand-50 ring-1 ring-brand-500"
                  : "border-slate-200 bg-white hover:border-slate-300"
              }`}
            >
              <input
                type="radio"
                name="decision"
                value={option.value}
                checked={decision === option.value}
                onChange={(e) => setDecision(e.target.value)}
                className="sr-only"
              />
              <span
                className={`mt-0.5 flex h-4 w-4 flex-none items-center justify-center rounded-full border-2 ${
                  decision === option.value ? "border-brand-600" : "border-slate-300"
                }`}
                aria-hidden="true"
              >
                {decision === option.value && <span className="h-2 w-2 rounded-full bg-brand-600" />}
              </span>
              <span>
                <span className="block font-semibold text-slate-900">{option.label}</span>
                <span className="mt-0.5 block text-xs text-slate-600">{option.description}</span>
              </span>
            </label>
          ))}
        </div>

        {decision === "override" && (
          <div>
            <span className="field-label">Replacement diagnosis</span>
            <p className="field-hint">
              Your replacement diagnosis will be recorded and made available as evidence for similar
              future cases — including, when this case had an image in a supported modality, for future
              cases with a visually similar image. It is shown to the model as a precedent to weigh,
              not as a guaranteed-correct answer.
            </p>
            {diagnoses.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {diagnoses.map((d, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => setOverrideDiagnosis(d.name)}
                    className="rounded-full border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-700 transition hover:border-brand-500 hover:text-brand-800"
                  >
                    {d.name}
                  </button>
                ))}
              </div>
            )}
            <input
              type="text"
              value={overrideDiagnosis}
              onChange={(e) => setOverrideDiagnosis(e.target.value)}
              placeholder="Pick a diagnosis above, or type your own"
              aria-label="Replacement diagnosis"
              className="field"
            />
          </div>
        )}

        {decision === "annotate" && (
          <div>
            <label className="field-label" htmlFor="annotation_note">
              Note
            </label>
            <textarea
              id="annotation_note"
              rows={3}
              value={annotationNote}
              onChange={(e) => setAnnotationNote(e.target.value)}
              placeholder="e.g. Recommend follow-up MRI in 3 months."
              className="field"
            />
          </div>
        )}

        <div>
          <label className="field-label" htmlFor="reviewer">
            Reviewer name
          </label>
          <input
            id="reviewer"
            type="text"
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
            placeholder="Dr. …"
            className="field"
          />
        </div>

        <div>
          <span className="field-label">
            Your confidence in this review <span className="font-normal text-slate-500">(optional)</span>
          </span>
          <div className="mt-1.5 flex items-center gap-1" role="radiogroup" aria-label="Reviewer confidence rating, 1 to 5">
            {[1, 2, 3, 4, 5].map((value) => (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={rating === value}
                aria-label={`${value} star${value === 1 ? "" : "s"}`}
                onClick={() => setRating(rating === value ? null : value)}
                className={`text-2xl leading-none transition ${
                  rating != null && value <= rating ? "text-amber-400" : "text-slate-300 hover:text-amber-300"
                }`}
              >
                ★
              </button>
            ))}
            {rating != null && (
              <button
                type="button"
                onClick={() => setRating(null)}
                className="ml-2 text-xs text-slate-500 hover:text-slate-800"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        <button type="submit" disabled={submitting} className="btn-primary w-full py-2.5">
          {submitting ? "Submitting…" : "Submit review"}
        </button>
      </form>
    </section>
  );

  return (
    <div className="animate-rise">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">Human in the loop</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">Review case</h2>
          <p className="mt-1 font-mono text-xs text-slate-500">Case {report.case_id}</p>
        </div>
        <button type="button" onClick={onBack} className="btn-secondary">
          <ArrowLeftIcon className="h-4 w-4" />
          Back to report
        </button>
      </div>

      <div className="mt-6">
        <ReportSummary report={report} imageSubmitted={Boolean(submittedSummary?.hadImage)} aside={decisionPanel} />
      </div>
    </div>
  );
}
