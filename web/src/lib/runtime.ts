/**
 * Reader of runtime.json (spec §9.6) as written by `mtmct.runtime.profiling.runtime_report`:
 * `{stages: {name: {wall_s, items, fps|items_per_s, p50_ms, p95_ms, cached}}, wall_time_s, fps, peak_rss_gb, mean_cpu_pct, cache_bytes}`.
 * Tolerates missing fields and a list of stages.
 */

export interface StageTiming {
  stage: string
  wallS: number
  p50Ms: number | null
  p95Ms: number | null
  /** Taken from the stage cache (wall time is near zero). */
  cached: boolean
}

export interface RuntimeSummary {
  wallS: number | null
  fps: number | null
  peakRssGb: number | null
  cpuMean: number | null
  cacheMb: number | null
  stages: StageTiming[]
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

function timing(stage: string, v: unknown): StageTiming | null {
  if (typeof v === 'number') return { stage, wallS: v, p50Ms: null, p95Ms: null, cached: false }
  if (!v || typeof v !== 'object') return null
  const o = v as Record<string, unknown>
  const wall = num(o.wall_s) ?? num(o.seconds) ?? num(o.time_s)
  return wall === null ? null : { stage, wallS: wall, p50Ms: num(o.p50_ms), p95Ms: num(o.p95_ms), cached: o.cached === true }
}

export function parseRuntime(rt: Record<string, unknown> | null | undefined): RuntimeSummary {
  const r = rt ?? {}
  const raw = r.stages
  const stages = (
    Array.isArray(raw)
      ? raw.map((s) => timing(String((s as Record<string, unknown>)?.stage ?? (s as Record<string, unknown>)?.name ?? '?'), s))
      : raw && typeof raw === 'object'
        ? Object.entries(raw).map(([k, v]) => timing(k, v))
        : []
  ).filter((s): s is StageTiming => s !== null)
  return {
    wallS: num(r.wall_time_s) ?? (stages.length ? stages.reduce((a, s) => a + s.wallS, 0) : null),
    fps: num(r.fps),
    peakRssGb: num(r.peak_rss_gb),
    cpuMean: num(r.mean_cpu_pct) ?? num(r.cpu_mean),
    cacheMb: num(r.cache_bytes) === null ? num(r.cache_mb) : num(r.cache_bytes)! / 2 ** 20,
    stages,
  }
}
