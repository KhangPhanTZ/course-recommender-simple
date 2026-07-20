import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api -> the FastAPI backend so the UI can call it
// without CORS headaches during local development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY || "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
