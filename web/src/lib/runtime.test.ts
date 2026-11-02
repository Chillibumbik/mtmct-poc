import { describe, expect, it } from 'vitest'
import { parseRuntime } from './runtime'

describe('parseRuntime', () => {
  it('reads runtime_report() output', () => {
    const r = parseRuntime({
      stages: {
        detect: { wall_s: 100, items: 2000, fps: 20, p50_ms: 60, p95_ms: 90, cached: false },
        reid: { wall_s: 0.01, items: 0, items_per_s: null, p50_ms: null, p95_ms: null, cached: true },
      },
      wall_time_s: 120,
      frames_processed: 2000,
      fps: 16.7,
      peak_rss_gb: 3.2,
      mean_cpu_pct: 85,
      cache_bytes: 3 * 2 ** 20,
    })
    expect(r.stages).toEqual([
      { stage: 'detect', wallS: 100, p50Ms: 60, p95Ms: 90, cached: false },
      { stage: 'reid', wallS: 0.01, p50Ms: null, p95Ms: null, cached: true },
    ])
    expect(r).toMatchObject({ wallS: 120, fps: 16.7, peakRssGb: 3.2, cpuMean: 85, cacheMb: 3 })
  })
  it('sums stage times when wall time is absent', () => {
    expect(parseRuntime({ stages: { a: { wall_s: 2 }, b: { wall_s: 3 } } }).wallS).toBe(5)
  })
  it('reads stage lists and bare numbers, ignores junk', () => {
    expect(parseRuntime({ stages: [{ stage: 'reid', seconds: 3 }, { name: 'x' }] }).stages.map((s) => s.stage)).toEqual(['reid'])
    expect(parseRuntime({ stages: { track: 7 } }).stages[0].wallS).toBe(7)
    expect(parseRuntime(null)).toMatchObject({ stages: [], wallS: null, fps: null })
  })
})
