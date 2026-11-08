import { useCallback, useEffect, useRef, useState } from "react";
import CaseSubmission from "./pages/CaseSubmission.jsx";
import Processing from "./pages/Processing.jsx";
import Report from "./pages/Report.jsx";
import Review from "./pages/Review.jsx";
import { diagnoseCase } from "./services/api.js";

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
  const lastSubmission = useRef(null);

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
      lastSubmission.current = submission;
      setError(null);
      setSubmittedSummary(submission.summary);
      setSubmitting(true);
      navigate(ROUTES.PROCESSING);
      try {
        const result = await diagnoseCase(submission.formData);
        setReport(result);
        setSubmitting(false);
        // Reachable automatically when review is required, not just via
        // the manual "Review Case" button on the Report screen. `replace`
        // so Back doesn't land on the (now meaningless) processing screen.
        navigate(result.review_required ? ROUTES.REVIEW : ROUTES.REPORT, { replace: true });
      } catch (err) {
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
    lastSubmission.current = null;
    setReport(null);
    setSubmittedSummary(null);
    setError(null);
    navigate(ROUTES.SUBMISSION);
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-6 py-4">
          <h1 className="text-xl font-semibold tracking-tight text-slate-800">MedOrchestrate</h1>
          <p className="text-sm text-slate-500">Clinical decision-support prototype — for clinician review only</p>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
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

        {screen === ROUTES.PROCESSING && <Processing />}

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
    </div>
  );
}
