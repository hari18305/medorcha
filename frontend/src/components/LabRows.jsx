import { MAX_LAB_ROWS } from "../utils/caseFields.js";
import { PlusIcon, XIcon } from "./icons.jsx";

let nextId = 0;
export const newLabRow = (name = "", value = "") => ({ id: `lab-new-${nextId++}`, name, value });

/**
 * Laboratory values as name/value rows. Units belong in the value ("10.2 g/dL"); the server keeps
 * values as written. Fully empty rows are ignored on submit; a half-filled row is rejected with a
 * message (see buildLabs in utils/caseFields.js).
 */
export default function LabRows({ rows, onRowsChange }) {
  const update = (id, patch) => onRowsChange(rows.map((r) => (r.id === id ? { ...r, ...patch } : r)));

  return (
    <div>
      <span className="field-label" id="labs-label">
        Laboratory values
      </span>
      {rows.length === 0 ? (
        <p className="field-hint">None added. Lab values count as clinical input for the analysis.</p>
      ) : (
        <ul className="mt-1.5 space-y-2" aria-labelledby="labs-label">
          {rows.map((row, index) => (
            <li key={row.id} className="flex items-center gap-2" data-lab-row>
              <input
                type="text"
                value={row.name}
                onChange={(e) => update(row.id, { name: e.target.value })}
                placeholder="hemoglobin"
                aria-label={`Lab ${index + 1} name`}
                className="field !mt-0 min-w-0 flex-1"
              />
              <input
                type="text"
                value={row.value}
                onChange={(e) => update(row.id, { value: e.target.value })}
                placeholder="10.2 g/dL"
                aria-label={`Lab ${index + 1} value`}
                className="field !mt-0 min-w-0 flex-1"
              />
              <button
                type="button"
                onClick={() => onRowsChange(rows.filter((r) => r.id !== row.id))}
                aria-label={`Remove lab ${index + 1}`}
                className="btn-ghost !px-2"
              >
                <XIcon className="h-4 w-4" />
              </button>
            </li>
          ))}
        </ul>
      )}
      <button
        type="button"
        onClick={() => onRowsChange([...rows, newLabRow()])}
        disabled={rows.length >= MAX_LAB_ROWS}
        className="btn-secondary mt-2 !px-3 !py-1.5 text-xs"
      >
        <PlusIcon className="h-3.5 w-3.5" />
        Add lab value
      </button>
    </div>
  );
}
