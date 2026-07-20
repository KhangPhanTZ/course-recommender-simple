/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // brand — teal (retrieval/vector world)
        brand: {
          50: "#ecfdf9",
          100: "#d0f7ee",
          200: "#a3efdd",
          300: "#6ee0c8",
          400: "#38c9ac",
          500: "#14b09a",
          600: "#0d8f83",
          700: "#0f716a",
          800: "#115a56",
          900: "#124a48",
          950: "#042c2c",
        },
        // accent — amber (the GenAI layer)
        amber: {
          400: "#f0b45a",
          500: "#e39a2f",
          600: "#c2761c",
        },
        // cool, slightly teal-biased neutrals
        ink: {
          50: "#f5f7f8",
          100: "#e9edf0",
          200: "#d3dae0",
          300: "#aeb9c2",
          400: "#7d8b96",
          500: "#5b6a75",
          600: "#48545e",
          700: "#37424c",
          800: "#232c34",
          900: "#161d23",
          950: "#0b0f13",
        },
      },
      fontFamily: {
        sans: [
          "InterVariable",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "JetBrains Mono",
          "Menlo",
          "monospace",
        ],
      },
      boxShadow: {
        card: "0 1px 2px rgba(16,24,32,.05), 0 12px 32px -16px rgba(16,24,32,.22)",
        glow: "0 0 0 1px rgba(20,176,154,.25), 0 16px 40px -12px rgba(20,176,154,.35)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "fade-up": "fade-up .6s cubic-bezier(.22,.61,.36,1) both",
      },
    },
  },
  plugins: [],
};
