import fs from 'node:fs'
import { defineConfig, devices } from '@playwright/test'

// Bundled Chromium has no H.264; prefer system Chrome when installed, otherwise the spec asserts on boxes, not video.
const chrome = fs.existsSync('/usr/bin/google-chrome')

export default defineConfig({
  testDir: './e2e',
  globalSetup: './e2e/global-setup.ts',
  timeout: 180_000,
  expect: { timeout: 15_000 },
  workers: 1,
  use: { baseURL: 'http://localhost:5173', trace: 'on-first-retry' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], ...(chrome ? { channel: 'chrome' } : {}) } }],
  webServer: { command: 'pnpm dev', url: 'http://localhost:5173', reuseExistingServer: true },
})
