/** What the canvas overlay draws for one camera at one instant (pure), and how. */
import { colorForId, textColorOn } from './colors'
import { diffStatus, frameAt, type CameraIndex } from './frames'

export type OverlayMode = 'pred' | 'gt' | 'diff'

export interface DrawBox {
  x: number
  y: number
  w: number
  h: number
  color: string
  label: string
  line: 'solid' | 'dashed' | 'dotted'
  weight: number
  /** Global id selected by a click (GT boxes: the id their person is mostly assigned). */
  gid: number | null
}

export const DIFF_COLORS = { ok: '#FFFFFF', error: '#D55E00', none: '#BBBBBB' } as const

export function overlayBoxes(idx: CameraIndex, t: number, mode: OverlayMode, majority: Map<number, number>): DrawBox[] {
  if (mode === 'pred') {
    const f = frameAt(idx.pred, t)
    return (f?.pred ?? []).map((b) => ({ ...b, color: colorForId(b.id), label: String(b.id), line: 'solid', weight: 2, gid: b.id }))
  }
  if (mode === 'gt') {
    const f = frameAt(idx.gt, t)
    return (f?.gt ?? []).map((b) => {
      const gid = majority.get(b.id) ?? null
      // coloured like the global id the person mostly gets, so a correct run looks the same in both modes
      return { ...b, color: colorForId(gid ?? b.id), label: `GT ${b.id}`, line: 'solid', weight: 2, gid }
    })
  }
  const f = frameAt(idx.pred, t)
  const gtFrame = f ?? frameAt(idx.gt, t)
  const matched = new Set(f?.match.filter((m) => m !== null))
  const out: DrawBox[] = (gtFrame?.gt ?? [])
    .filter((b) => !matched.has(b.id))
    .map((b) => ({ ...b, color: DIFF_COLORS.none, label: `GT ${b.id}: пропуск`, line: 'dotted', weight: 1, gid: majority.get(b.id) ?? null }))
  f?.pred.forEach((b, i) => {
    const s = diffStatus(b.id, f.match[i], majority)
    if (s.kind === 'ok') out.push({ ...b, color: DIFF_COLORS.ok, label: String(b.id), line: 'solid', weight: 1, gid: b.id })
    else if (s.kind === 'unmatched') out.push({ ...b, color: DIFF_COLORS.none, label: `${b.id} · нет GT`, line: 'dashed', weight: 2, gid: b.id })
    else out.push({ ...b, color: DIFF_COLORS.error, label: `${b.id} ≠ ${s.expected} (GT ${s.gt})`, line: 'solid', weight: 3, gid: b.id })
  })
  return out
}

/** Topmost-looking (smallest) box under a point in original-resolution pixels. */
export function hitTest(boxes: DrawBox[], x: number, y: number): DrawBox | null {
  let best: DrawBox | null = null
  for (const b of boxes)
    if (b.gid !== null && x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h && (!best || b.w * b.h < best.w * best.h)) best = b
  return best
}

/** Draws boxes scaled from original pixels (`sx`, `sy` = canvas px per original px). */
export function drawBoxes(ctx: CanvasRenderingContext2D, boxes: DrawBox[], sx: number, sy: number, dpr: number, selected: number | null) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
  const font = Math.round(11 * dpr)
  ctx.font = `600 ${font}px "Geist Variable", system-ui, sans-serif`
  ctx.textBaseline = 'top'
  // selected person last, so it is on top
  const ordered = selected === null ? boxes : [...boxes.filter((b) => b.gid !== selected), ...boxes.filter((b) => b.gid === selected)]
  for (const b of ordered) {
    const isSel = selected !== null && b.gid === selected
    ctx.globalAlpha = selected === null || isSel ? 1 : 0.3
    const x = b.x * sx
    const y = b.y * sy
    const w = b.w * sx
    const h = b.h * sy
    const lw = (b.weight + (isSel ? 1.5 : 0)) * dpr
    // dark halo keeps light colours visible on bright frames
    ctx.setLineDash(b.line === 'dashed' ? [6 * dpr, 4 * dpr] : b.line === 'dotted' ? [2 * dpr, 3 * dpr] : [])
    ctx.lineWidth = lw + 2 * dpr
    ctx.strokeStyle = 'rgba(0,0,0,0.45)'
    ctx.strokeRect(x, y, w, h)
    ctx.lineWidth = lw
    ctx.strokeStyle = b.color
    ctx.strokeRect(x, y, w, h)
    ctx.setLineDash([])
    const pad = 3 * dpr
    const tw = ctx.measureText(b.label).width + 2 * pad
    const th = font + 2 * pad
    // above the box, kept inside the canvas
    const ly = y - th >= 0 ? y - th : Math.max(0, y)
    const lx = Math.min(Math.max(0, x - lw / 2), ctx.canvas.width - tw)
    ctx.fillStyle = b.color
    ctx.fillRect(lx, ly, tw, th)
    ctx.fillStyle = textColorOn(b.color)
    ctx.fillText(b.label, lx + pad, ly + pad)
  }
  ctx.globalAlpha = 1
}
