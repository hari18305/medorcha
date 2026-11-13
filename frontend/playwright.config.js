import { defineConfig } from "@playwright/test";

// Browser end-to-end tests: the REAL built frontend against the REAL FastAPI backend with every model and network call faked
// (scripts/e2e_backend.py). They use the Chrome that is already installed (channel "chrome"), so no browser is downloaded.
// It serves the frontend on port 3000 (an origin the backend's CORS list already allows) so it does not collide with `npm run dev` on 5173.
// Run: npm run test:e2e   (from frontend/). Set E2E_PYTHON to the Python that has the project's requirements.
const python = process.env.E2E_PYTHON || (process.platform === "win32" ? "venv\\Scripts\\python.exe" : "venv/bin/python");

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: { baseURL: "http://localhost:3000", channel: "chrome", headless: true, trace: "retain-on-failure" },
  webServer: [
    {
      command: `${python} scripts${process.platform === "win32" ? "\\" : "/"}e2e_backend.py`,
      cwd: "..",
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: "npm run build && npx vite preview --port 3000 --strictPort",
      url: "http://localhost:3000",
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
