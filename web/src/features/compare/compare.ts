import type { CompareRow, RunOut } from '@/api/types.gen'
import { formatMetric } from '@/lib/metrics'

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

/** Config keys (dotted) whose value differs between the compared runs; a key absent from `config_diff` = profile default. */
export function varyingKeys(runs: RunOut[]): string[] {
  const keys = new Set(runs.flatMap((r) => Object.keys(r.config_diff)))
  return [...keys].filter((k) => runs.some((r) => !same(r.config_diff[k], runs[0].config_diff[k]))).sort()
}

/** Signed delta in the metric's display units (percentage points for ratios). */
export function formatDelta(metric: string, d: number | null | undefined): string {
  if (d === null || d === undefined || !Number.isFinite(d)) return ''
  if (Math.abs(d) < 1e-9) return '±0'
  return `${d > 0 ? '+' : '−'}${formatMetric(metric, Math.abs(d))}`
}

export type Tone = 'good' | 'bad' | 'neutral'

export function deltaTone(d: number | null | undefined, higherIsBetter: boolean): Tone {
  if (d === null || d === undefined || Math.abs(d) < 1e-9) return 'neutral'
  return d > 0 === higherIsBetter ? 'good' : 'bad'
}

/** Runs holding the best value of a row (all of them on a tie); empty when there is nothing to prefer. */
export function bestRuns(row: CompareRow): Set<number> {
  const vals = Object.entries(row.values).filter((e): e is [string, number] => e[1] !== null && e[1] !== undefined)
  if (vals.length < 2) return new Set()
  const best = (row.higher_is_better ? Math.max : Math.min)(...vals.map(([, v]) => v))
  if (vals.every(([, v]) => Math.abs(v - best) < 1e-9)) return new Set()
  return new Set(vals.filter(([, v]) => Math.abs(v - best) < 1e-9).map(([id]) => Number(id)))
}

/** Short run label for chart axes / legends. */
export function runLabel(r: RunOut): string {
  return `#${r.id} ${r.name.length > 22 ? `${r.name.slice(0, 21)}…` : r.name}`
}
