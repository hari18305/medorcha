import { useCallback, useEffect, useRef, useState } from "react";
import CaseSubmission from "./pages/CaseSubmission.jsx";
import Processing from "./pages/Processing.jsx";
import Report from "./pages/Report.jsx";
import Review from "./pages/Review.jsx";
import { ApiError, getDiagnosisStatus, startDiagnosis } from "./services/api.js";
import { followAnalysis } from "./utils/progress.js";
import { BrandMark, CheckIcon, ShieldIcon } from "./components/icons.jsx";

// Client-side routing (Phase 10) without a router dependency: each screen
// has a hash URL (#/, #/processing, #/report, #/review), so the browser's
// Back/Forward buttons work and a screen can be linked to. Hash routes
// need no server-side fallback, so this works on any static host
// (`serve`, nginx, a CDN) unchanged.
//
// All CASE data still lives in React state only -- never localStorage or
// sessionStorage (Phase 10 requirement). So a page refresh drops the
// in-memory report, and the guards below send the user back to the
// submission screen instead of rendering a report that no longer exists.
const ROUTES = {
  SUBMISSION: "/",
  PROCESSING: "/processing",
  REPORT: "/report",
  REVIEW: "/review",
};
const KNOWN_PATHS = Object.values(ROUTES);

function currentPath() {
  const path = window.location.hash.replace(/^#/, "");
  return KNOWN_PATHS.includes(path) ? path : ROUTES.SUBMISSION;
}

function useHashRoute() {
  const [path, setPath] = useState(currentPath);

  useEffect(() => {
    const onHashChange = () => setPath(currentPath());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const navigate = useCallback((to, { replace = false } = {}) => {
    if (replace) {
      // replaceState doesn't fire hashchange, so update state directly.
      window.history.replaceState(null, "", `#${to}`);
      setPath(to);
    } else {
      window.location.hash = to; // pushes a history entry and fires hashchange
    }
  }, []);

  return [path, navigate];
}

export default function App() {
  const [path, navigate] = useHashRoute();
  const [submitting, setSubmitting] = useState(false);
  const [report, setReport] = useState(null);
  const [submittedSummary, setSubmittedSummary] = useState(null);
  const [error, setError] = useState(null); // an ApiError (or Error) from the last submission
  const [progress, setProgress] = useState(null); // latest /diagnose/{id}/status body while analysing
  const lastSubmission = useRef(null);
  // Incremented whenever a run is started or abandoned, so a polling loop that outlived its screen
  // (the user started a new case) stops instead of overwriting newer state.
  const runToken = useRef(0);

  // Guards: /processing only makes sense mid-request; /report and /review
  // need a report in memory. Anything else falls back to the form.
  const invalid =
    (path === ROUTES.PROCESSING && !submitting) ||
    ((path === ROUTES.REPORT || path === ROUTES.REVIEW) && !report);
  const screen = invalid ? ROUTES.SUBMISSION : path;

  useEffect(() => {
    if (invalid) navigate(ROUTES.SUBMISSION, { replace: true });
  }, [invalid, navigate]);

  const runSubmission = useCallback(
    async (submission) => {
      const token = ++runToken.current;
      lastSubmission.current = submission;
      setError(null);
      setProgress(null);
      setSubmittedSummary(submission.summary);
      setSubmitting(true);
      navigate(ROUTES.PROCESSING);
      try {
        const result = await followAnalysis({
          start: () => startDiagnosis(submission.formData),
          fetchStatus: getDiagnosisStatus,
          onProgress: (status) => {
            if (runToken.current === token) setProgress(status);
          },
          shouldStop: () => runToken.current !== token,
          makeError: (message) => new ApiError(message, { kind: "server" }),
        });
        if (result === null || runToken.current !== token) return; // abandoned
        setReport(result);
        setSubmitting(false);
        // Reachable automatically when review is required, not just via
        // the manual "Review Case" button on the Report screen. `replace`
        // so Back doesn't land on the (now meaningless) processing screen.
        navigate(result.review_required ? ROUTES.REVIEW : ROUTES.REPORT, { replace: true });
      } catch (err) {
        if (runToken.current !== token) return;
        setError(err);
        setSubmitting(false);
        navigate(ROUTES.SUBMISSION, { replace: true });
      }
    },
    [navigate]
  );

  function handleSubmit(formData, summary) {
    runSubmission({ formData, summary });
  }

  function handleRetry() {
    if (lastSubmission.current) runSubmission(lastSubmission.current);
  }

  function handleNewCase() {
    runToken.current += 1; // abandon any analysis still being followed
    setSubmitting(false);
    setProgress(null);
    lastSubmission.current = null;
    setReport(null);
    setSubmittedSummary(null);
    setError(null);
    navigate(ROUTES.SUBMISSION);
  }

  return (
    <div className="flex min-h-screen flex-col bg-canvas text-slate-900">
      <header className="bg-ink-900 text-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-6 gap-y-3 px-5 py-3.5 sm:px-8">
          <button
            type="button"
            onClick={handleNewCase}
            className="flex items-center gap-3 rounded-lg text-left focus-visible:ring-offset-ink-900"
            aria-label="MedOrchestrate, start a new case"
          >
            <BrandMark className="h-9 w-9" />
            <span>
              <span className="block text-base font-semibold leading-tight tracking-tight">MedOrchestrate</span>
              <span className="block text-xs leading-tight text-slate-400">
                Evidence-grounded multimodal decision support
              </span>
            </span>
          </button>
          <Stepper screen={screen} hasReport={Boolean(report)} />
        </div>
        <div className="border-t border-white/10 bg-ink-800">
          <p className="mx-auto flex max-w-6xl items-center gap-2 px-5 py-1.5 text-xs text-slate-300 sm:px-8">
            <ShieldIcon className="h-3.5 w-3.5 text-brand-300" />
            Research prototype — supports, never replaces, clinician judgement. Not a medical device.
          </p>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-5 py-8 sm:px-8 sm:py-10">
        {screen === ROUTES.SUBMISSION && (
          <CaseSubmission
            onSubmit={handleSubmit}
            error={error}
            onRetry={lastSubmission.current ? handleRetry : undefined}
            // keep what was typed if a submission failed, so a 422 can be
            // fixed (or a network blip retried) without re-entering the case
            initialValues={submittedSummary}
          />
        )}

        {screen === ROUTES.PROCESSING && <Processing progress={progress} />}

        {screen === ROUTES.REPORT && report && (
          <Report
            report={report}
            submittedSummary={submittedSummary}
            onNewCase={handleNewCase}
            onReviewCase={() => navigate(ROUTES.REVIEW)}
          />
        )}

        {screen === ROUTES.REVIEW && report && (
          <Review
            report={report}
            submittedSummary={submittedSummary}
            onBack={() => navigate(ROUTES.REPORT)}
            onReviewed={() => navigate(ROUTES.REPORT)}
          />
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <p className="mx-auto max-w-6xl px-5 py-4 text-xs text-slate-500 sm:px-8">
          Confidence is a heuristic blend of retrieval strength, live-evidence level and source agreement. It is
          not a probability that a diagnosis is correct. In the browser, case data lives only in this tab's memory and is not saved
          to local storage.
        </p>
      </footer>
    </div>
  );
}

// Progress through the four screens. Purely presentational: it mirrors the
// current route and never navigates (the guards above own navigation).
const STEPS = [
  { key: ROUTES.SUBMISSION, label: "Case" },
  { key: ROUTES.PROCESSING, label: "Analysis" },
  { key: ROUTES.REPORT, label: "Report" },
  { key: ROUTES.REVIEW, label: "Review" },
];

function Stepper({ screen, hasReport }) {
  const current = STEPS.findIndex((step) => step.key === screen);
  return (
    <nav aria-label="Progress">
      <ol className="flex items-center gap-1 text-xs font-medium">
        {STEPS.map((step, index) => {
          const done = index < current || (hasReport && index < 2 && index !== current);
          const active = index === current;
          return (
            <li key={step.key} className="flex items-center gap-1" aria-current={active ? "step" : undefined}>
              <span
                className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 ${
                  active
                    ? "bg-white text-ink-900"
                    : done
                      ? "text-brand-200"
                      : "text-slate-500"
                }`}
              >
                <span
                  className={`flex h-4 w-4 items-center justify-center rounded-full text-[10px] ${
                    active ? "bg-brand-700 text-white" : done ? "bg-brand-700/60 text-white" : "bg-white/10"
                  }`}
                >
                  {done && !active ? <CheckIcon className="h-2.5 w-2.5" /> : index + 1}
                </span>
                {/* labels of inactive steps collapse on phones so the header fits */}
                <span className={active ? "" : "hidden sm:inline"}>{step.label}</span>
              </span>
              {index < STEPS.length - 1 && <span className="h-px w-3 bg-white/15" aria-hidden="true" />}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
