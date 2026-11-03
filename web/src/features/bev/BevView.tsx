import { useEffect, useMemo, useRef } from 'react'
import { Map as MapIcon } from 'lucide-react'
import { useRunTrajectories } from '@/api/hooks'
import type { TrajectoriesOut } from '@/api/types.gen'
import { EmptyState, QueryView } from '@/components/app/States'
import { useChartTheme, type ChartTheme } from '@/lib/chart'
import type { MasterClock } from '@/features/player/clock'
import { colorForId, textColorOn } from '@/features/player/colors'
import { nearestIndex } from '@/features/player/frames'
import { fitBounds, niceStep } from './bev'

/** A person's marker is shown if a trajectory point lies within this many seconds (API buckets are 0.5 s). */
const MARKER_TOL_S = 0.75
const TAIL_S = 3
/** Gaps longer than this break the polyline (person left all cameras). */
const BREAK_S = 2
const PAD = 28

interface Track {
  gid: number
  ts: number[]
  xy: [number, number][]
}

function BevCanvas({ data, clock, selected, onSelect }: { data: TrajectoriesOut; clock: MasterClock; selected: number | null; onSelect: (gid: number) => void }) {
  const theme = useChartTheme()
  const tracks = useMemo<Track[]>(
    () => data.trajectories.map((t) => ({ gid: t.global_id, ts: t.points.map((p) => p.ts), xy: t.points.map((p) => [p.x, p.y]) })),
    [data],
  )
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const latest = useRef({ tracks, theme, selected })
  const dirty = useRef(true)
  const markers = useRef<{ gid: number; x: number; y: number }[]>([])
  useEffect(() => {
    latest.current = { tracks, theme, selected }
    dirty.current = true
  })

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')
    const bounds = data.bounds!
    let raf = 0
    let lastT = NaN
    const draw = () => {
      const t = clock.time()
      if (!ctx || (!dirty.current && t === lastT)) return
      dirty.current = false
      lastT = t
      render(ctx, canvas, bounds, t, latest.current.tracks, latest.current.theme, latest.current.selected, markers.current)
    }
    const loop = () => {
      raf = requestAnimationFrame(loop)
      draw()
    }
    raf = requestAnimationFrame(loop)
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.round(canvas.clientWidth * dpr)
      canvas.height = Math.round(canvas.clientHeight * dpr)
      dirty.current = true
      draw()
    })
    ro.observe(canvas)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [clock, data])

  const [x0, y0, x1, y1] = data.bounds!
  const ratio = Math.min(1.2, Math.max(0.45, (y1 - y0 + 1) / (x1 - x0 + 1)))
  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-lg border bg-card"
      style={{ aspectRatio: `1 / ${ratio}`, maxHeight: 'calc(100dvh - 260px)' }}
      aria-label="Вид сверху: мировые траектории людей и их положение в текущий момент"
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect()
        const dpr = window.devicePixelRatio || 1
        const px = (e.clientX - r.left) * dpr
        const py = (e.clientY - r.top) * dpr
        let best: { gid: number; d: number } | null = null
        for (const m of markers.current) {
          const d = Math.hypot(m.x - px, m.y - py)
          if (d < 16 * dpr && (!best || d < best.d)) best = { gid: m.gid, d }
        }
        if (best) onSelect(best.gid)
      }}
    />
  )
}

function render(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  bounds: readonly [number, number, number, number],
  t: number,
  tracks: Track[],
  theme: ChartTheme,
  selected: number | null,
  markers: { gid: number; x: number; y: number }[],
) {
  const dpr = window.devicePixelRatio || 1
  const W = canvas.width
  const H = canvas.height
  ctx.clearRect(0, 0, W, H)
  const { toCanvas } = fitBounds(bounds, W, H, PAD * dpr)
  const [x0, y0, x1, y1] = bounds

  // metric grid
  const step = niceStep(Math.max(x1 - x0, y1 - y0))
  ctx.lineWidth = 1
  ctx.strokeStyle = theme.grid
  ctx.fillStyle = theme.muted
  ctx.font = `${Math.round(10 * dpr)}px "Geist Variable", system-ui, sans-serif`
  for (let x = Math.ceil(x0 / step) * step; x <= x1 + 1e-9; x += step) {
    const [cx] = toCanvas(x, y0)
    ctx.beginPath()
    ctx.moveTo(cx, 0)
    ctx.lineTo(cx, H)
    ctx.stroke()
    ctx.fillText(`${+x.toFixed(2)} м`, cx + 3 * dpr, H - 4 * dpr)
  }
  for (let y = Math.ceil(y0 / step) * step; y <= y1 + 1e-9; y += step) {
    const [, cy] = toCanvas(x0, y)
    ctx.beginPath()
    ctx.moveTo(0, cy)
    ctx.lineTo(W, cy)
    ctx.stroke()
    ctx.fillText(`${+y.toFixed(2)}`, 3 * dpr, cy - 3 * dpr)
  }

  const path = (tr: Track, from: number, to: number) => {
    ctx.beginPath()
    for (let i = from; i < to; i++) {
      const [cx, cy] = toCanvas(tr.xy[i][0], tr.xy[i][1])
      if (i === from || tr.ts[i] - tr.ts[i - 1] > BREAK_S) ctx.moveTo(cx, cy)
      else ctx.lineTo(cx, cy)
    }
    ctx.stroke()
  }
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  // full trajectories, faint
  for (const tr of tracks) {
    const sel = selected === tr.gid
    ctx.globalAlpha = selected === null ? 0.35 : sel ? 0.9 : 0.1
    ctx.strokeStyle = colorForId(tr.gid)
    ctx.lineWidth = (sel ? 2.5 : 1.5) * dpr
    path(tr, 0, tr.ts.length)
  }
  // last seconds + current position
  markers.length = 0
  const font = Math.round(11 * dpr)
  ctx.font = `600 ${font}px "Geist Variable", system-ui, sans-serif`
  for (const tr of tracks) {
    const i = nearestIndex(tr.ts, t)
    if (i < 0 || Math.abs(tr.ts[i] - t) > MARKER_TOL_S) continue
    const color = colorForId(tr.gid)
    ctx.globalAlpha = selected === null || selected === tr.gid ? 1 : 0.3
    ctx.strokeStyle = color
    ctx.lineWidth = 3 * dpr
    const from = Math.max(0, nearestIndex(tr.ts, t - TAIL_S))
    path(tr, from, i + 1)
    const [cx, cy] = toCanvas(tr.xy[i][0], tr.xy[i][1])
    ctx.beginPath()
    ctx.arc(cx, cy, (selected === tr.gid ? 7 : 5) * dpr, 0, 2 * Math.PI)
    ctx.fillStyle = color
    ctx.fill()
    ctx.lineWidth = 1.5 * dpr
    ctx.strokeStyle = theme.surface
    ctx.stroke()
    const label = String(tr.gid)
    const pad = 3 * dpr
    const tw = ctx.measureText(label).width + 2 * pad
    ctx.fillStyle = color
    ctx.fillRect(cx + 7 * dpr, cy - font - pad, tw, font + 2 * pad)
    ctx.fillStyle = textColorOn(color)
    ctx.textBaseline = 'top'
    ctx.fillText(label, cx + 7 * dpr + pad, cy - font)
    ctx.textBaseline = 'alphabetic'
    markers.push({ gid: tr.gid, x: cx, y: cy })
  }
  ctx.globalAlpha = 1
}

/** BEV tab of the player: world trajectories (`/runs/{id}/trajectories`) with markers at the master-clock time. */
export function BevView({ runId, clock, selected, onSelect }: { runId: number; clock: MasterClock; selected: number | null; onSelect: (gid: number) => void }) {
  const q = useRunTrajectories(runId)
  return (
    <QueryView
      query={q}
      isEmpty={(d) => !d.available || !d.bounds || d.trajectories.length === 0}
      empty={
        <EmptyState
          icon={<MapIcon className="size-6" />}
          title={q.data?.available === false ? 'Вид сверху недоступен' : 'Нет мировых координат'}
          hint={
            q.data?.available === false
              ? 'У датасета нет калибровки камер (гомографий на плоскость пола), поэтому мировые траектории не вычисляются.'
              : 'В результатах прогона нет точек с мировыми координатами.'
          }
        />
      }
    >
      {(d) => (
        <div className="space-y-1.5">
          <BevCanvas data={d} clock={clock} selected={selected} onSelect={onSelect} />
          <p className="text-xs text-muted-foreground">
            Вид сверху, метры. Линии — траектории по глобальным ID (точка на 0,5 с), жирный хвост — последние {TAIL_S} с, кружок — положение
            в текущий момент. Клик по кружку — маршрут человека.
          </p>
        </div>
      )}
    </QueryView>
  )
}
