import { expect, test } from '@playwright/test'

test('overview renders', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Обзор' })).toBeVisible()
})
