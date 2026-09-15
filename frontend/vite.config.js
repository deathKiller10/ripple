import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Builds to frontend/dist, which ripple/server.py serves automatically when
// present. Until then the server falls back to frontend/dashboard.html, so the
// container always has a working UI even with no Node toolchain.
export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
