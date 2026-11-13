import { mergeHistory, parseHistoryInput } from "../utils/caseFields.js";
import { PlusIcon, XIcon } from "./icons.jsx";

// One-click shortcuts. They add nothing until clicked and are plain history terms, not suggestions
// about this patient.
const QUICK_ADD = ["hypertension", "diabetes", "atrial fibrillation", "on anticoagulation", "smoker"];

/**
 * Past-history chips. `items` is the committed list, `draft` the text still in the box.
 * Both live in the parent so that text typed but not yet "added" is still submitted
 * (see finalizeHistory in utils/caseFields.js) instead of silently dropped.
 *
 * Enter, comma and semicolon commit the draft; pasting a comma-separated list adds each term.
 * Enter is intercepted so it never submits the surrounding form.
 */
export default function HistoryInput({ items, onItemsChange, draft, onDraftChange, id = "history" }) {
  function commitDraft(text = draft) {
    const additions = parseHistoryInput(text);
    if (additions.length) onItemsChange(mergeHistory(items, additions));
    onDraftChange("");
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" || event.key === "," || event.key === ";") {
      event.preventDefault();
      commitDraft();
    } else if (event.key === "Backspace" && draft === "" && items.length > 0) {
      onItemsChange(items.slice(0, -1));
    }
  }

  function handlePaste(event) {
    const text = event.clipboardData?.getData("text") ?? "";
    if (/[,;\n]/.test(text)) {
      event.preventDefault();
      commitDraft(draft + text);
    }
  }

  const remaining = QUICK_ADD.filter((q) => !items.some((i) => i.toLowerCase() === q));

  return (
    <div>
      <label className="field-label" htmlFor={id}>
        Past medical history
      </label>
      <div className="field mt-1.5 flex min-h-[2.5rem] flex-wrap items-center gap-1.5 !py-1.5 focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-500/25">
        {items.map((item) => (
          <span
            key={item}
            data-history-chip={item}
            className="inline-flex items-center gap-1 rounded-full bg-brand-50 py-0.5 pl-2.5 pr-1 text-sm text-brand-900 ring-1 ring-inset ring-brand-200"
          >
            {item}
            <button
              type="button"
              onClick={() => onItemsChange(items.filter((i) => i !== item))}
              aria-label={`Remove ${item}`}
              className="rounded-full p-0.5 text-brand-700 hover:bg-brand-100"
            >
              <XIcon className="h-3.5 w-3.5" />
            </button>
          </span>
        ))}
        <input
          id={id}
          type="text"
          value={draft}
          onChange={(e) => onDraftChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          onBlur={() => commitDraft()}
          placeholder={items.length ? "" : "e.g. hypertension, then Enter"}
          className="min-w-[8rem] flex-1 border-0 bg-transparent p-0.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-0"
        />
      </div>
      <p className="field-hint">Press Enter or comma to add. Conditions and relevant treatments (e.g. “on warfarin”).</p>
      {remaining.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-slate-500">Quick add:</span>
          {remaining.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => onItemsChange(mergeHistory(items, [q]))}
              className="inline-flex items-center gap-1 rounded-full border border-slate-300 bg-white px-2 py-0.5 text-xs text-slate-600 transition hover:border-brand-500 hover:text-brand-700"
            >
              <PlusIcon className="h-3 w-3" />
              {q}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
