/**
 * Renders a visible, distinct message per failure type (Phase 10: "visible
 * error states for network/422/500 failures").
 *
 * `error` is whatever was caught: an ApiError (see services/api.js, has
 * `.kind` / `.status`), a plain string (a client-side form-validation
 * message), or any other Error (treated as an unexpected failure).
 * `onRetry`, if given, shows a "Try again" action -- only offered for
 * failures that can plausibly succeed on a retry (network / server), never
 * for a 422, where the input itself has to change first.
 */

const RETRYABLE_KINDS = new Set(["network", "server"]);

function describe(error) {
  if (typeof error === "string") {
    return { title: "Please check the form", description: error, kind: "validation" };
  }
  switch (error?.kind) {
    case "network":
      return {
        title: "Can't reach the server",
        description: error.message,
        kind: "network",
      };
    case "validation":
      return {
        title: "The server rejected this submission",
        description: error.message,
        kind: "validation",
      };
    case "not_found":
      return { title: "Not found", description: error.message, kind: "not_found" };
    case "server":
      return {
        title: "The server ran into a problem",
        description:
          "Something went wrong while processing this request on the server. Please try again in a moment.",
        kind: "server",
      };
    default:
      return {
        title: "Something went wrong",
        description: error?.message || "An unexpected error occurred.",
        kind: "other",
      };
  }
}

export default function ErrorBanner({ error, onRetry }) {
  if (!error) return null;
  const { title, description, kind } = describe(error);

  return (
    <div role="alert" data-error-kind={kind} className="rounded-md border border-red-200 bg-red-50 px-4 py-3">
      <p className="text-sm font-semibold text-red-800">{title}</p>
      <p className="mt-1 text-sm text-red-700">{description}</p>
      {onRetry && RETRYABLE_KINDS.has(kind) && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 text-sm font-medium text-red-800 underline underline-offset-2 hover:text-red-900"
        >
          Try again
        </button>
      )}
    </div>
  );
}
