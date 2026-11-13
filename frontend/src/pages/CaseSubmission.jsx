import { useRef, useState } from "react";
import ErrorBanner from "../components/ErrorBanner.jsx";
import {
  AlertIcon,
  ArrowRightIcon,
  BookIcon,
  DocIcon,
  GlobeIcon,
  ImageIcon,
  UploadIcon,
  UserCheckIcon,
  XIcon,
} from "../components/icons.jsx";

const SEX_OPTIONS = ["male", "female", "other"];

// One-click demo cases. These are the project's own text-only test cases
// (data/test_cases/*), so what appears here is what the evaluation ran.
const EXAMPLES = [
  {
    label: "Hypertension, headache",
    age: "45",
    sex: "male",
    clinicalText: "headache and dizziness, history of hypertension",
  },
  {
    label: "Elderly, on warfarin",
    age: "81",
    sex: "male",
    clinicalText:
      "gradual gait unsteadiness and mild headache over several weeks, on warfarin, history of a fall two months ago; history of hypertension and atrial fibrillation",
  },
  {
    label: "Positional vertigo",
    age: "57",
    sex: "male",
    clinicalText: "brief episodes of spinning vertigo triggered by rolling over in bed, lasting under a minute, no headache",
  },
];

// What the pipeline does with a case, in order. Kept in step with the real
// agents (clinical, imaging, rag, evidence, fusion, optimizer) and the real
// sources they query.
const PIPELINE = [
  { icon: DocIcon, title: "Structure the case", text: "Symptoms, history and labs are extracted from your text and report." },
  { icon: ImageIcon, title: "Read the image", text: "BiomedCLIP gives a fast zero-shot read of CT, MRI or chest X-ray." },
  { icon: BookIcon, title: "Retrieve offline evidence", text: "A curated MedlinePlus and PubMed knowledge base, plus past clinician precedents." },
  { icon: GlobeIcon, title: "Check live literature", text: "PubMed, Europe PMC, ClinicalTrials.gov and MedlinePlus, queried per case." },
  { icon: UserCheckIcon, title: "Score, refine, hand over", text: "Confidence below 65% triggers query refinement, then mandatory clinician review." },
];

/**
 * onSubmit(formData, submittedSummary) -- formData is ready to POST to
 * /diagnose as multipart/form-data; submittedSummary is a plain object
 * ({age, sex, clinicalText, medicalReport, hadImage, imageFile}) the
 * parent keeps in React state so the Report screen can show a patient
 * summary, and so this form can be re-populated after a failed
 * submission. FinalReport itself doesn't echo back patient details, so
 * the app remembers what was submitted instead of inventing data that
 * isn't in the API response.
 *
 * error: whatever the last submission threw (an ApiError) -- rendered by
 * ErrorBanner with a distinct message per failure type. onRetry re-sends
 * the same submission (offered for network/server failures only).
 * initialValues: the previously submitted summary, if any.
 */
export default function CaseSubmission({ onSubmit, error, onRetry, initialValues }) {
  const [age, setAge] = useState(initialValues?.age ?? "");
  const [sex, setSex] = useState(initialValues?.sex ?? "");
  const [clinicalText, setClinicalText] = useState(initialValues?.clinicalText ?? "");
  const [medicalReport, setMedicalReport] = useState(initialValues?.medicalReport ?? "");
  const [imageFile, setImageFile] = useState(initialValues?.imageFile ?? null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(() =>
    initialValues?.imageFile ? URL.createObjectURL(initialValues.imageFile) : null
  );
  const [formError, setFormError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  function setFile(file) {
    setImageFile(file);
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
    setImagePreviewUrl(file ? URL.createObjectURL(file) : null);
  }

  function handleImageChange(event) {
    setFile(event.target.files?.[0] ?? null);
  }

  function handleRemoveImage() {
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) setFile(file);
  }

  function applyExample(example) {
    setAge(example.age);
    setSex(example.sex);
    setClinicalText(example.clinicalText);
    setMedicalReport("");
    setFormError(null);
  }

  function handleSubmit(event) {
    event.preventDefault();
    setFormError(null);

    if (!age || !sex) {
      setFormError("Age and sex are required.");
      return;
    }
    if (!clinicalText.trim() && !medicalReport.trim() && !imageFile) {
      setFormError("Provide at least clinical text, a medical report, or an image.");
      return;
    }

    const formData = new FormData();
    formData.append("age", age);
    formData.append("sex", sex);
    if (clinicalText.trim()) formData.append("clinical_text", clinicalText.trim());
    if (medicalReport.trim()) formData.append("medical_report", medicalReport.trim());
    if (imageFile) formData.append("image", imageFile);

    onSubmit(formData, {
      age,
      sex,
      clinicalText: clinicalText.trim(),
      medicalReport: medicalReport.trim(),
      hadImage: Boolean(imageFile),
      imageFile,
    });
  }

  // A client-side form message (string) takes precedence over the last
  // server error until the user submits again.
  const displayedError = formError || error;

  // Mirrors the backend's input-sufficiency gate (confidence.py): with no
  // clinical text and no report there is nothing to extract symptoms or
  // findings from, so confidence is capped at 50% and review is required.
  const mayBeInsufficient = Boolean(imageFile) && !clinicalText.trim() && !medicalReport.trim();

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div className="animate-rise">
        <p className="eyebrow">New case</p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">Describe the patient</h2>
        <p className="mt-2 max-w-xl text-sm text-slate-600">
          Enter whatever is available: clinical text, a report, an image, or any combination. The more clinical
          context you give, the more the result can be grounded in evidence.
        </p>

        {displayedError && (
          <div className="mt-5">
            <ErrorBanner error={displayedError} onRetry={formError ? undefined : onRetry} />
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-6 space-y-5">
          <section className="card card-pad">
            <h3 className="text-sm font-semibold text-slate-900">Patient</h3>
            <div className="mt-3 grid grid-cols-2 gap-4">
              <div>
                <label className="field-label" htmlFor="age">
                  Age
                </label>
                <input
                  id="age"
                  type="number"
                  min="0"
                  max="120"
                  inputMode="numeric"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  className="field"
                />
              </div>
              <div>
                <label className="field-label" htmlFor="sex">
                  Sex
                </label>
                <select id="sex" value={sex} onChange={(e) => setSex(e.target.value)} className="field">
                  <option value="">Select…</option>
                  {SEX_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option.charAt(0).toUpperCase() + option.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </section>

          <section className="card card-pad">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-slate-900">Clinical information</h3>
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-slate-500">Try an example:</span>
                {EXAMPLES.map((example) => (
                  <button
                    key={example.label}
                    type="button"
                    onClick={() => applyExample(example)}
                    className="rounded-full border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 transition hover:border-brand-500 hover:text-brand-700"
                  >
                    {example.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-4">
              <label className="field-label" htmlFor="clinical_text">
                Presenting complaint and history
              </label>
              <textarea
                id="clinical_text"
                rows={4}
                value={clinicalText}
                onChange={(e) => setClinicalText(e.target.value)}
                placeholder="e.g. headache and dizziness, history of hypertension"
                className="field"
              />
            </div>

            <div className="mt-4">
              <label className="field-label" htmlFor="medical_report">
                Medical or imaging report text
              </label>
              <textarea
                id="medical_report"
                rows={4}
                value={medicalReport}
                onChange={(e) => setMedicalReport(e.target.value)}
                placeholder="Paste relevant report text here, if available"
                className="field"
              />
            </div>
          </section>

          <section className="card card-pad">
            <h3 className="text-sm font-semibold text-slate-900">
              Medical image <span className="font-normal text-slate-500">(optional)</span>
            </h3>
            <p className="field-hint">CT head, brain MRI or chest X-ray. Other modalities are not assessed.</p>

            <label
              htmlFor="image"
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={`mt-3 flex cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed px-4 py-6 text-center transition ${
                dragging
                  ? "border-brand-500 bg-brand-50"
                  : "border-slate-300 bg-slate-50/60 hover:border-brand-400 hover:bg-brand-50/50"
              }`}
            >
              <UploadIcon className="h-6 w-6 text-slate-400" />
              <span className="text-sm font-medium text-slate-700">Drop an image here, or click to browse</span>
              <span className="text-xs text-slate-500">PNG or JPEG</span>
            </label>
            <input
              id="image"
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="sr-only"
            />

            {imagePreviewUrl && (
              <div className="mt-3 flex items-center gap-3 rounded-lg border border-slate-200 p-2.5">
                <img
                  src={imagePreviewUrl}
                  alt="Selected medical image preview"
                  className="h-16 w-16 rounded-md border border-slate-200 object-cover"
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-slate-800">{imageFile?.name}</p>
                  <p className="text-xs text-slate-500">
                    {imageFile ? `${Math.max(1, Math.round(imageFile.size / 1024))} KB` : ""}
                  </p>
                </div>
                <button type="button" onClick={handleRemoveImage} className="btn-ghost" aria-label="Remove image">
                  <XIcon className="h-4 w-4" />
                  Remove
                </button>
              </div>
            )}
          </section>

          {mayBeInsufficient && (
            <div
              role="status"
              data-testid="sparse-input-hint"
              className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3"
            >
              <AlertIcon className="mt-0.5 h-4 w-4 text-amber-700" />
              <p className="text-sm text-amber-900">
                <span className="font-semibold">Image only.</span> With no symptoms or report findings to reason
                from, confidence is capped at 50% and this case will be sent for clinician review.
              </p>
            </div>
          )}

          <button type="submit" className="btn-primary w-full py-3 text-base">
            Analyze case
            <ArrowRightIcon className="h-4 w-4" />
          </button>
        </form>
      </div>

      <aside className="animate-rise lg:pt-[3.75rem]" style={{ animationDelay: "80ms" }}>
        <div className="card card-pad lg:sticky lg:top-6">
          <p className="eyebrow">What happens next</p>
          <ol className="mt-4 space-y-4">
            {PIPELINE.map(({ icon: PipelineIcon, title, text }, index) => (
              <li key={title} className="flex gap-3">
                <span className="relative flex h-8 w-8 flex-none items-center justify-center rounded-lg bg-brand-50 text-brand-700 ring-1 ring-inset ring-brand-100">
                  <PipelineIcon className="h-4 w-4" />
                  <span className="absolute -left-1.5 -top-1.5 flex h-4 w-4 items-center justify-center rounded-full bg-ink-800 text-[9px] font-semibold text-white">
                    {index + 1}
                  </span>
                </span>
                <div>
                  <p className="text-sm font-semibold text-slate-900">{title}</p>
                  <p className="mt-0.5 text-xs leading-relaxed text-slate-600">{text}</p>
                </div>
              </li>
            ))}
          </ol>
          <p className="mt-5 border-t border-slate-100 pt-4 text-xs leading-relaxed text-slate-500">
            Every diagnosis in the report cites the evidence behind it, and low-confidence results always go to a
            clinician before anyone acts on them.
          </p>
        </div>
      </aside>
    </div>
  );
}
