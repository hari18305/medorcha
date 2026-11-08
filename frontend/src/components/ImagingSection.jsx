import ConfidenceBar from "./ConfidenceBar.jsx";

/**
 * Imaging findings block of the clinician report (Phase 10).
 *
 * `imaging`: the FinalReport.imaging object ({modality, findings[],
 * confidence, unsupported_modality}) or null/undefined.
 * `imageSubmitted`: whether the user attached an image on the submission
 * screen -- lets this distinguish "no image was provided" (render nothing)
 * from "an image was provided but no findings came back" (say so).
 *
 * An unsupported modality (one with no BiomedCLIP findings model) is
 * rendered honestly as NOT ASSESSED -- never as if the placeholder
 * "no findings model available..." string were a real finding. This
 * mirrors how the backend treats it downstream: the RAG query and the
 * fusion prompt both ignore imaging for such a case.
 */
export default function ImagingSection({ imaging, imageSubmitted = false }) {
  if (!imaging && !imageSubmitted) return null;

  return (
    <section className="mt-6" data-section="imaging">
      <h3 className="text-sm font-semibold text-slate-700">Imaging findings</h3>
      <div className="mt-3 rounded-lg border border-slate-200 bg-white p-4">
        {!imaging ? (
          <p className="text-sm italic text-slate-500">
            An image was submitted, but no imaging findings were recorded for this case (the imaging
            analysis may have failed). The differential below does not include any imaging evidence.
          </p>
        ) : imaging.unsupported_modality ? (
          <div data-imaging-state="not-assessed">
            <p className="text-sm text-slate-700">
              <span className="font-medium">Modality:</span> {imaging.modality}
            </p>
            <p className="mt-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
              <span className="font-semibold">Not assessed.</span> This prototype has no findings model
              for this modality. Treat this case as if no imaging had been provided — nothing from the
              image was used in the differential or the evidence search.
            </p>
          </div>
        ) : (
          <div data-imaging-state="assessed">
            <p className="text-sm text-slate-700">
              <span className="font-medium">Modality:</span> {imaging.modality}
            </p>
            {imaging.findings?.length > 0 && (
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
                {imaging.findings.map((finding, index) => (
                  <li key={index}>{finding}</li>
                ))}
              </ul>
            )}
            <div className="mt-3 max-w-xs">
              <ConfidenceBar confidence={imaging.confidence} size="sm" label="BiomedCLIP confidence" />
            </div>
            <p className="mt-3 text-xs text-slate-400">
              A fast zero-shot pre-filter over a small fixed label set for this modality — a sanity
              check, not a diagnostic read of the image.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
