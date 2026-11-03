import { useEffect, useRef, useState } from 'react'
import { VideoOff } from 'lucide-react'
import { cn } from '@/lib/utils'
import { syncVideo, videoTime, type MasterClock } from './clock'
import type { CameraIndex } from './frames'
import { drawBoxes, hitTest, overlayBoxes, type DrawBox, type OverlayMode } from './overlay'

interface Props {
  camera: string
  src: string | null
  poster: string | null
  index: CameraIndex | undefined
  clock: MasterClock
  mode: OverlayMode
  majority: Map<number, number>
  selected: number | null
  focused: boolean
  onSelect: (gid: number) => void
}

type VideoStatus = 'loading' | 'ok' | 'error' | 'none'

/**
 * One camera: <video> slaved to the master clock + canvas overlay. While playing, boxes are drawn from
 * `requestVideoFrameCallback` at the exact presented frame time; otherwise (paused, no video, no rVFC) from rAF.
 */
export function CameraTile(props: Props) {
  const { camera, src, poster, index, clock, focused } = props
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const latest = useRef(props)
  const dirty = useRef(true)
  const drawn = useRef<DrawBox[]>([])
  const videoOk = useRef(false)
  const [status, setStatus] = useState<VideoStatus>(src ? 'loading' : 'none')
  const [inRange, setInRange] = useState(true)

  useEffect(() => {
    latest.current = props
    dirty.current = true
  })

  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')
    let raf = 0
    let vfc = 0
    let lastT = NaN
    let rangeShown = true

    const draw = (t: number) => {
      if (!ctx || (!dirty.current && t === lastT)) return
      dirty.current = false
      lastT = t
      const p = latest.current
      if (!p.index) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        drawn.current = []
        return
      }
      const boxes = overlayBoxes(p.index, t, p.mode, p.majority)
      drawn.current = boxes
      const { width, height } = p.index.meta
      drawBoxes(ctx, boxes, canvas.width / width, canvas.height / height, window.devicePixelRatio || 1, p.selected)
    }

    const hasVfc = !!video && 'requestVideoFrameCallback' in video
    const onFrame = (_: number, meta: VideoFrameCallbackMetadata) => {
      const idx = latest.current.index
      if (idx && clock.playing) draw(meta.mediaTime + idx.meta.frame_offset / idx.meta.fps)
      vfc = video!.requestVideoFrameCallback(onFrame)
    }
    if (hasVfc) vfc = video.requestVideoFrameCallback(onFrame)

    const loop = () => {
      raf = requestAnimationFrame(loop)
      const idx = latest.current.index
      const t = clock.time()
      let ok = true
      if (video && videoOk.current && idx) {
        const m = idx.meta
        const a = syncVideo(videoTime(t, m.frame_offset, m.fps), video, m.fps, clock.playing, clock.speed)
        ok = a.inRange
        if (a.seekTo !== null) video.currentTime = a.seekTo
        if (Math.abs(video.playbackRate - a.playbackRate) > 1e-3) video.playbackRate = a.playbackRate
        const play = clock.playing && a.inRange
        if (play && video.paused) video.play().catch(() => undefined)
        else if (!play && !video.paused) video.pause()
      }
      if (ok !== rangeShown) setInRange((rangeShown = ok))
      const byVfc = hasVfc && videoOk.current && clock.playing && ok && !video.paused
      if (!byVfc) draw(t)
    }
    raf = requestAnimationFrame(loop)

    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr))
      canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr))
      dirty.current = true
      draw(video && videoOk.current && latest.current.index ? video.currentTime + latest.current.index.meta.frame_offset / latest.current.index.meta.fps : clock.time())
    })
    ro.observe(canvas)
    return () => {
      cancelAnimationFrame(raf)
      if (hasVfc) video.cancelVideoFrameCallback(vfc)
      ro.disconnect()
    }
  }, [clock])

  const toOriginal = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const m = latest.current.index?.meta
    return m ? hitTest(drawn.current, ((e.clientX - r.left) / r.width) * m.width, ((e.clientY - r.top) / r.height) * m.height) : null
  }

  const aspect = index ? `${index.meta.width} / ${index.meta.height}` : '4 / 3'
  return (
    <div className={cn('relative overflow-hidden rounded-md bg-black', focused && 'ring-2 ring-brand ring-offset-2 ring-offset-background')} style={{ aspectRatio: aspect }}>
      {src && (
        <video
          ref={videoRef}
          src={src}
          poster={poster ?? undefined}
          muted
          playsInline
          preload="auto"
          className="absolute inset-0 size-full object-fill"
          onLoadedMetadata={() => {
            videoOk.current = true
            dirty.current = true
            setStatus('ok')
          }}
          onError={() => {
            videoOk.current = false
            setStatus('error')
          }}
        />
      )}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 size-full"
        aria-label={`Камера ${camera}: рамки и глобальные ID`}
        onClick={(e) => {
          const b = toOriginal(e)
          if (b?.gid != null) props.onSelect(b.gid)
        }}
        onMouseMove={(e) => (e.currentTarget.style.cursor = toOriginal(e) ? 'pointer' : 'default')}
      />
      <div className="pointer-events-none absolute bottom-1.5 left-1.5 rounded bg-black/65 px-1.5 py-0.5 font-mono text-[11px] text-white">{camera}</div>
      {(status === 'error' || status === 'none') && (
        <div className="pointer-events-none absolute right-1.5 bottom-1.5 flex items-center gap-1 rounded bg-black/65 px-1.5 py-0.5 text-[11px] text-white/80">
          <VideoOff className="size-3" /> нет видео
        </div>
      )}
      {status === 'ok' && !inRange && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center bg-black/55 text-xs text-white/80">камера вне диапазона времени</div>
      )}
      {!index && <div className="pointer-events-none absolute inset-x-0 top-0 h-0.5 animate-pulse bg-brand/70" aria-label="Загрузка рамок" />}
    </div>
  )
}
