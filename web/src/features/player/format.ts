/** Common-clock time as `m:ss.cc`. */
export function formatClock(t: number): string {
  const s = Math.max(0, t)
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`
}
