/**
 * Small fetch wrapper for the MedOrchestrate FastAPI backend (Phase 9).
 *
 * Every function throws an ApiError on failure. `error.message` is always
 * a clinician-readable message; `error.kind` says WHAT failed so the UI
 * can show a distinct message per failure type (see ErrorBanner.jsx):
 *
 *   "network"    -- fetch itself rejected (server unreachable/offline/CORS);
 *                   `status` is undefined
 *   "validation" -- HTTP 422, the server rejected the request as invalid
 *   "not_found"  -- HTTP 404
 *   "server"     -- HTTP 5xx
 *   "other"      -- any other non-2xx status
 */

// Build-time override (Vite convention: only VITE_-prefixed vars are
// exposed to client code). Falls back to localhost:8000 for local dev
// (`npm run dev`) and for Docker Compose, since the browser reaches the
// backend via its published host port either way -- not through the
// compose-internal service network, which the browser can't resolve.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const NETWORK_ERROR_MESSAGE =
  "Could not reach the MedOrchestrate server. Check your connection and that the backend is running.";

export class ApiError extends Error {
  constructor(message, { kind, status } = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind ?? "other";
    this.status = status;
  }
}

export function kindForStatus(status) {
  if (status === 422) return "validation";
  if (status === 404) return "not_found";
  if (status >= 500) return "server";
  return "other";
}

function formatDetail(detail) {
  if (typeof detail === "string") return detail;
  // FastAPI's default 422 shape is a list of {loc, msg, type} objects.
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.at(-1) : undefined;
        return field ? `${field}: ${item.msg}` : item.msg;
      })
      .join("; ");
  }
  return JSON.stringify(detail);
}

async function handleResponse(response) {
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) {
        message = formatDetail(body.detail);
      }
    } catch {
      // Response body wasn't JSON (or was empty) -- fall back to the
      // generic status-based message above.
    }
    throw new ApiError(message, { kind: kindForStatus(response.status), status: response.status });
  }
  return response.json();
}

async function safeFetch(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new ApiError(NETWORK_ERROR_MESSAGE, { kind: "network" });
  }
  return handleResponse(response);
}

/** POST /diagnose -- formData must already be a FormData instance. */
export function diagnoseCase(formData) {
  return safeFetch(`${API_BASE_URL}/diagnose`, {
    method: "POST",
    body: formData,
  });
}

/** GET /diagnose/{case_id} */
export function getDiagnosis(caseId) {
  return safeFetch(`${API_BASE_URL}/diagnose/${encodeURIComponent(caseId)}`);
}

/** POST /review/{case_id} -- review = { decision, annotation, reviewer, rating } -- rating is optional, 1-5 or null. */
export function submitReview(caseId, review) {
  return safeFetch(`${API_BASE_URL}/review/${encodeURIComponent(caseId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(review),
  });
}
