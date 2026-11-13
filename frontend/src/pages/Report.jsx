import ReportSummary from "../components/ReportSummary.jsx";
import { PlusIcon, UserCheckIcon } from "../components/icons.jsx";

/**
 * report: the FinalReport JSON from POST/GET /diagnose (includes the
 * imaging findings, if any, as report.imaging).
 * submittedSummary: { age, sex, clinicalText, medicalReport, history, labs, hadImage }
 * echoed from what was submitted on the Case Submission screen --
 * FinalReport doesn't include patient details, so this app remembers what
 * was submitted rather than inventing data that isn't in the API response.
 * onNewCase / onReviewCase: navigation callbacks from App.jsx.
 */
export default function Report({ report, submittedSummary, onNewCase, onReviewCase }) {
  const labEntries = Object.entries(submittedSummary?.labs ?? {});
  const patient = submittedSummary && (
    <section className="card card-pad">
      <h3 className="text-sm font-semibold text-slate-900">Patient summary</h3>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 text-sm">
        <div>
          <dt className="text-xs text-slate-500">Age</dt>
          <dd className="font-medium text-slate-900">{submittedSummary.age || "—"}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500">Sex</dt>
          <dd className="font-medium capitalize text-slate-900">{submittedSummary.sex || "—"}</dd>
        </div>
      </dl>
      {submittedSummary.clinicalText && (
        <div className="mt-3 border-t border-slate-100 pt-3">
          <p className="eyebrow">Presenting complaint</p>
          <p className="mt-1 whitespace-pre-wrap text-sm text-slate-700">{submittedSummary.clinicalText}</p>
        </div>
      )}
      {submittedSummary.history?.length > 0 && (
        <div className="mt-3 border-t border-slate-100 pt-3" data-summary="history">
          <p className="eyebrow">Past history</p>
          <ul className="mt-1.5 flex flex-wrap gap-1.5">
            {submittedSummary.history.map((item) => (
              <li key={item} className="rounded-full bg-slate-100 px-2.5 py-0.5 text-sm text-slate-700">
                {item}
              </li>
            ))}
          </ul>
        </div>
      )}
      {labEntries.length > 0 && (
        <div className="mt-3 border-t border-slate-100 pt-3" data-summary="labs">
          <p className="eyebrow">Laboratory values</p>
          <dl className="mt-1.5 divide-y divide-slate-100 text-sm">
            {labEntries.map(([name, value]) => (
              <div key={name} className="flex items-baseline justify-between gap-4 py-1">
                <dt className="text-slate-500">{name}</dt>
                <dd className="text-right font-medium tabular-nums text-slate-900">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
      {submittedSummary.medicalReport && (
        <div className="mt-3 border-t border-slate-100 pt-3">
          <p className="eyebrow">Reported medical text</p>
          <p className="mt-1 whitespace-pre-wrap text-sm text-slate-700">{submittedSummary.medicalReport}</p>
        </div>
      )}
    </section>
  );

  return (
    <div className="animate-rise">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">Clinician report</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">Decision-support report</h2>
          <p className="mt-1 font-mono text-xs text-slate-500">Case {report.case_id}</p>
        </div>
        <div className="flex gap-2">
          <button type="button" onClick={onReviewCase} className="btn-primary">
            <UserCheckIcon className="h-4 w-4" />
            Review case
          </button>
          <button type="button" onClick={onNewCase} className="btn-secondary">
            <PlusIcon className="h-4 w-4" />
            New case
          </button>
        </div>
      </div>

      <div className="mt-6">
        <ReportSummary report={report} imageSubmitted={Boolean(submittedSummary?.hadImage)} aside={patient} />
      </div>
    </div>
  );
}
