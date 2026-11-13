import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
  // Component tests (React Testing Library in jsdom). The plain-JS helper tests in src/utils/*.test.js
  // use node:test and run separately (`npm run test:unit`), so vitest only picks up *.test.jsx.
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.jsx"],
    setupFiles: ["./src/test/setup.js"],
    css: false,
  },
});
