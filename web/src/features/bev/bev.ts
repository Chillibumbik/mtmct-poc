/** World (metres, y up) → canvas (px, y down) fit, preserving aspect ratio. */
export function fitBounds(bounds: readonly [number, number, number, number], width: number, height: number, pad: number) {
  const [x0, y0, x1, y1] = bounds
  const ew = Math.max(x1 - x0, 1e-6)
  const eh = Math.max(y1 - y0, 1e-6)
  const s = Math.min((width - 2 * pad) / ew, (height - 2 * pad) / eh)
  const ox = (width - ew * s) / 2
  const oy = (height - eh * s) / 2
  return {
    scale: s,
    toCanvas: (x: number, y: number): [number, number] => [ox + (x - x0) * s, height - oy - (y - y0) * s],
  }
}

/** 1/2/5·10ⁿ grid step giving about `target` lines over `extent`. */
export function niceStep(extent: number, target = 8): number {
  const raw = Math.max(extent, 1e-9) / target
  const p = 10 ** Math.floor(Math.log10(raw))
  const m = raw / p
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p
}
