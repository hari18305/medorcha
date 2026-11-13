// Pure helpers for the structured case fields (past history, lab values).
//
// The backend contract (backend/api/routes.py): `history` is a JSON array of strings and
// `labs` a JSON object of name -> value. The server merges both into the clinical findings
// (clinical_agent.merge_structured_fields): history is de-duplicated case-insensitively and
// labs override anything inferred from prose. These helpers keep the form's own input clean
// and bounded; they never decide anything clinical.

export const MAX_HISTORY_ITEMS = 20;
export const MAX_HISTORY_LENGTH = 120;
export const MAX_LAB_ROWS = 30;
export const MAX_LAB_FIELD_LENGTH = 60;

const norm = (s) => s.toLowerCase().replace(/\s+/g, " ").trim();

/** "hypertension, diabetes; smoker\nCKD" -> ["hypertension", "diabetes", "smoker", "CKD"] */
export function parseHistoryInput(text) {
  return String(text ?? "")
    .split(/[,;\n]/)
    .map((part) => part.replace(/\s+/g, " ").trim())
    .filter(Boolean);
}

/** Append `additions` to `existing`, skipping case-insensitive duplicates; first spelling wins. */
export function mergeHistory(existing, additions) {
  const out = [...existing];
  const seen = new Set(out.map(norm));
  for (const item of additions) {
    const key = norm(item);
    if (key && !seen.has(key)) {
      seen.add(key);
      out.push(item.replace(/\s+/g, " ").trim());
    }
  }
  return out;
}

/** The history list to send: the chips plus whatever is still typed in the box. Returns {history, error}. */
export function finalizeHistory(chips, draft) {
  const history = mergeHistory(chips, parseHistoryInput(draft));
  if (history.length > MAX_HISTORY_ITEMS) {
    return { history, error: `Past history is limited to ${MAX_HISTORY_ITEMS} items.` };
  }
  const tooLong = history.find((h) => h.length > MAX_HISTORY_LENGTH);
  if (tooLong) {
    return { history, error: `A history item is longer than ${MAX_HISTORY_LENGTH} characters: "${tooLong.slice(0, 30)}…"` };
  }
  return { history, error: null };
}

/**
 * Rows of {name, value} -> {labs, error}. Fully empty rows are ignored; a row with only one
 * of the two fields, a repeated name (case-insensitive) or an over-long field is an error.
 * Values stay strings exactly as typed ("10.2 g/dL"): units are part of the value.
 */
export function buildLabs(rows) {
  const labs = {};
  const seen = new Set();
  let used = 0;
  for (const row of rows) {
    const name = String(row.name ?? "").trim();
    const value = String(row.value ?? "").trim();
    if (!name && !value) continue;
    if (!name || !value) {
      return { labs: {}, error: `Each lab needs both a name and a value (${name || value} is missing its ${name ? "value" : "name"}).` };
    }
    if (name.length > MAX_LAB_FIELD_LENGTH || value.length > MAX_LAB_FIELD_LENGTH) {
      return { labs: {}, error: `Lab names and values are limited to ${MAX_LAB_FIELD_LENGTH} characters.` };
    }
    const key = norm(name);
    if (seen.has(key)) {
      return { labs: {}, error: `The lab "${name}" is listed twice.` };
    }
    seen.add(key);
    if (++used > MAX_LAB_ROWS) {
      return { labs: {}, error: `Lab values are limited to ${MAX_LAB_ROWS} rows.` };
    }
    labs[name] = value;
  }
  return { labs, error: null };
}

/** Turn a stored labs object back into editable rows (for restoring a failed submission). */
export function labsToRows(labs) {
  return Object.entries(labs ?? {}).map(([name, value], i) => ({ id: `lab-${i}`, name, value: String(value) }));
}
