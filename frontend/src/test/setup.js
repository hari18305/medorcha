import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// jsdom has no object URLs, no layout and no scrolling; the components use all three.
let urlCounter = 0;
URL.createObjectURL = vi.fn(() => `blob:test-${urlCounter++}`);
URL.revokeObjectURL = vi.fn();
Element.prototype.scrollIntoView = vi.fn();
window.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 0);

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  window.location.hash = "";
});
