import { defineConfig, devices } from '@playwright/test'

// E2E scenarios land in T11; browsers: `pnpm exec playwright install chromium`.
export default defineConfig({
  testDir: './e2e',
  use: { baseURL: 'http://localhost:5173', trace: 'on-first-retry' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: { command: 'pnpm dev', url: 'http://localhost:5173', reuseExistingServer: true },
})
