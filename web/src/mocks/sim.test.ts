import { describe, expect, it } from 'vitest'
import { METRIC_SECTIONS } from '@/lib/metrics'
import { db } from './sim'

describe('mock backend fixtures', () => {
  it('every finished run has a complete metrics.json (spec §9.7)', () => {
    const done = db.runs.filter((r) => r.status === 'done')
    expect(done.length).toBeGreaterThanOrEqual(8)
    for (const run of done) {
      const m = db.runData.get(run.id)!.metrics as Record<string, Record<string, unknown>>
      for (const cam of Object.values(m.per_camera)) expect(Object.keys(cam as object)).toEqual([...METRIC_SECTIONS.per_camera])
      expect(Object.keys(m.cross_camera)).toEqual([...METRIC_SECTIONS.cross_camera])
      expect(Object.keys(m.identity)).toEqual([...METRIC_SECTIONS.identity])
      expect(Object.keys(m.reid)).toEqual([...METRIC_SECTIONS.reid])
      const curve = m.threshold_curve as unknown as Record<string, number>[]
      expect(Object.keys(curve[0])).toEqual([...METRIC_SECTIONS.threshold_curve])
      expect(run.key_metrics.cross_hota).not.toBeNull()
    }
  })

  it('includes overlap and non-overlap datasets and one running job', () => {
    expect(new Set(db.datasets.map((d) => d.overlap))).toEqual(new Set(['overlap', 'nonoverlap']))
    expect(db.jobs.filter((j) => j.status === 'running')).toHaveLength(1)
  })
})
