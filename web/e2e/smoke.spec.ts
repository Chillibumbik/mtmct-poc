import { expect, test } from '@playwright/test'

// tsconfig.node has no DOM lib: describe the canvas structurally
type Canvas2D = { width: number; height: number; getContext(k: '2d'): { getImageData(x: number, y: number, w: number, h: number): { data: Uint8ClampedArray } } }

test('data → oracle run → jobs → metrics → player with boxes', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Обзор' })).toBeVisible()

  // Данные: epfl_terrace is ready
  await page.getByRole('link', { name: 'Данные' }).click()
  const card = page.locator('[data-slot="card"]', { hasText: 'epfl_terrace' }).first()
  await expect(card.getByText('готов', { exact: true })).toBeVisible()

  // Новый прогон: oracle, 20 s
  await page.getByRole('link', { name: 'Новый прогон' }).click()
  await expect(page.getByRole('heading', { name: 'Новый прогон' })).toBeVisible()
  await page.getByRole('radio', { name: 'oracle' }).click()
  await page.getByPlaceholder('весь').fill('20')
  const created = page.waitForResponse((r) => r.url().endsWith('/api/runs') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Запустить', exact: true }).click()
  const run = (await (await created).json()) as { id: number; job_id: number }

  // Задачи: the job shows up and finishes
  await expect(page).toHaveURL(/\/jobs/)
  const row = page.getByRole('row').filter({ hasText: `${run.id}` }).first()
  await expect(row).toBeVisible()
  await expect(row.getByText('готово', { exact: true })).toBeVisible({ timeout: 150_000 })

  // Карточка прогона: метрики
  await page.goto(`/runs/${run.id}`)
  await expect(page.getByText('IDF1').first()).toBeVisible()
  const metrics = await page.request.get(`/api/runs/${run.id}/metrics`)
  expect((await metrics.json()).metrics.cross_camera.IDF1).toBeGreaterThanOrEqual(0.9)

  // Плеер: boxes are requested and drawn on the canvas at a moment when someone is in c0
  const tracks = await page.request.get(`/api/runs/${run.id}/tracks?camera=c0&t0=0&t1=20`)
  const withBoxes = ((await tracks.json()).frames as { ts: number; pred: unknown[] }[]).find((f) => f.pred.length > 0)
  expect(withBoxes, 'oracle run has boxes in c0').toBeDefined()
  await page.getByRole('link', { name: 'Плеер' }).click()
  await expect(page).toHaveURL(/\/player/)
  await page.goto(`/runs/${run.id}/player?t=${withBoxes!.ts}`)
  const canvas = page.getByLabel(/Камера c0: рамки/)
  await expect(canvas).toBeVisible()
  await expect
    .poll(() =>
      canvas.evaluate((el) => {
        const c = el as unknown as Canvas2D
        const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
        for (let i = 3; i < d.length; i += 4) if (d[i] !== 0) return true
        return false
      }),
    )
    .toBe(true)
})
