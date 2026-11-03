/** Time ranges: what part of the sequence a run covers, and which box windows to fetch around the playhead. */
import type { SequenceOut } from '@/api/types.gen'
import { getPath } from '@/lib/config'

/** Window length for `/runs/{id}/tracks` (the API serves up to 30 s per call, whatever the fps). */
export const WINDOW_S = 8

/** Same rule as `mtmct.pipeline.run.time_range`: split range, shifted by `data.t_start`, cut to `data.max_seconds`. */
export function runTimeRange(config: Record<string, unknown> | null | undefined, seq: SequenceOut, split: string): [number, number] {
  const [a, b] = seq.splits[split] ?? [0, 1]
  const t0 = a * seq.duration_s + Number(getPath(config ?? {}, 'data.t_start') ?? 0)
  const max = getPath(config ?? {}, 'data.max_seconds')
  const t1 = b * seq.duration_s
  return [t0, typeof max === 'number' ? Math.min(t1, t0 + max) : t1]
}

/** Windows (index k covers [k·W, (k+1)·W)) needed at time t: current, next, and one more at ≥ 2× speed. */
export function windowsAt(t: number, speed: number, range: [number, number]): number[] {
  const k = Math.floor(t / WINDOW_S)
  const ks = speed >= 2 ? [k, k + 1, k + 2] : [k, k + 1]
  return ks.filter((i) => i >= 0 && i * WINDOW_S < range[1] && (i + 1) * WINDOW_S > range[0])
}
