import { describe, expect, it } from 'vitest'
import { KEY_METRIC_KEYS, METRIC_SECTIONS, formatMetric, metricDef } from './metrics'

// Every metric key of spec §9.7 (metrics.json), §9.1 (per-camera list) and §9.3 (identity extras).
const SPEC_KEYS: Record<string, string[]> = {
  per_camera: ['HOTA', 'DetA', 'AssA', 'LocA', 'IDF1', 'IDP', 'IDR', 'MOTA', 'MOTP', 'IDSW', 'FP', 'FN', 'Frag'],
  cross_camera: ['HOTA', 'AssA', 'IDF1', 'IDP', 'IDR', 'MOTA'],
  identity: [
    'pairwise_p',
    'pairwise_r',
    'pairwise_f1',
    'bcubed_p',
    'bcubed_r',
    'bcubed_f',
    'purity',
    'inverse_purity',
    'n_pred_ids',
    'n_gt_ids',
    'false_merges',
    'false_splits',
    'handover_success',
  ],
  reid: ['rank1', 'rank5', 'mAP'],
  threshold_curve: ['threshold', 'pairwise_p', 'pairwise_r', 'IDF1'],
}

describe('metric definitions', () => {
  for (const [section, keys] of Object.entries(SPEC_KEYS)) {
    it.each(keys)(`${section}.%s has label, Russian description and direction`, (key) => {
      const def = metricDef(key)
      expect(def, key).toBeDefined()
      expect(def!.label.length).toBeGreaterThan(0)
      expect(def!.description).toMatch(/[а-яё]/i)
      expect([true, false, null]).toContain(def!.higherIsBetter)
      expect(METRIC_SECTIONS[section as keyof typeof METRIC_SECTIONS]).toContain(key)
    })
  }

  it('covers every KeyMetrics field shown in tables', () => {
    for (const key of KEY_METRIC_KEYS) expect(metricDef(key), key).toBeDefined()
  })

  it('direction matches metric semantics', () => {
    expect(metricDef('IDF1')!.higherIsBetter).toBe(true)
    expect(metricDef('IDSW')!.higherIsBetter).toBe(false)
    expect(metricDef('false_merges')!.higherIsBetter).toBe(false)
    expect(metricDef('n_gt_ids')!.higherIsBetter).toBeNull()
  })
})

describe('formatMetric', () => {
  it('formats ratios as percent points', () => {
    expect(formatMetric('IDF1', 0.6234)).toBe('62.3')
    expect(formatMetric('MOTA', -0.125)).toBe('−12.5')
  })
  it('formats counts, resources and thresholds', () => {
    expect(formatMetric('IDSW', 1234)).toBe('1 234')
    expect(formatMetric('peak_rss_gb', 3.456)).toBe('3.46 ГБ')
    expect(formatMetric('fps', 12.34)).toBe('12.3')
    expect(formatMetric('wall_time_s', 125)).toBe('2 мин 05 с')
    expect(formatMetric('threshold', 0.45)).toBe('0.45')
  })
  it('renders missing values as a dash', () => {
    expect(formatMetric('HOTA', null)).toBe('—')
    expect(formatMetric('HOTA', undefined)).toBe('—')
    expect(formatMetric('HOTA', Number.NaN)).toBe('—')
  })
  it('falls back gracefully for unknown keys', () => {
    expect(formatMetric('unknown_metric', 0.5)).toBe('0.500')
    expect(formatMetric('unknown_count', 7)).toBe('7')
  })
})
