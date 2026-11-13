/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Primary: a deep clinical teal. Used for actions, focus and the brand mark.
        brand: {
          50: "#effafa",
          100: "#d5f2f1",
          200: "#aee5e4",
          300: "#7bd1d1",
          400: "#46b6b8",
          500: "#299a9e",
          600: "#1e7d84",
          700: "#1c646b",
          800: "#1c5157",
          900: "#1b4448",
        },
        // Chrome (header, dark panels): a blue-black ink rather than pure slate.
        ink: {
          700: "#243746",
          800: "#172736",
          900: "#0e1b27",
        },
        // Page and surface tones.
        canvas: "#f3f6f8",
      },
      fontFamily: {
        // System stacks only: a clinical tool should not call out to a font CDN.
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(14, 27, 39, 0.05), 0 1px 3px rgba(14, 27, 39, 0.06)",
        lift: "0 4px 12px rgba(14, 27, 39, 0.08), 0 2px 4px rgba(14, 27, 39, 0.05)",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        flash: {
          "0%": { boxShadow: "0 0 0 0 rgba(41, 154, 158, 0.55)" },
          "100%": { boxShadow: "0 0 0 10px rgba(41, 154, 158, 0)" },
        },
      },
      animation: {
        rise: "rise 0.35s ease-out both",
        flash: "flash 1.1s ease-out 1",
      },
    },
  },
  plugins: [],
};
