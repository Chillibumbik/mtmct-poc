/**
 * One master clock for all cameras (seconds on the common clock `ts = (frame_id + frame_offset) / fps`) and the
 * rule that keeps each <video> on it. The clock runs on wall time, so cameras without video (or with stalls) never
 * hold the others back; videos follow it.
 */

export type Now = () => number

const NOTIFY_MS = 100

export interface ClockSnapshot {
  t: number
  playing: boolean
  speed: number
}

export class MasterClock {
  readonly min: number
  readonly max: number
  playing = false
  speed = 1
  private base: number
  private anchor = 0
  private lastNotify = -Infinity
  private readonly now: Now
  private readonly listeners = new Set<() => void>()

  constructor(min: number, max: number, start = min, now: Now = () => performance.now()) {
    this.min = min
    this.max = Math.max(min, max)
    this.now = now
    this.base = this.clamp(start)
    this.snapshot = { t: this.base, playing: false, speed: 1 }
  }

  /** Last notified state, for `useSyncExternalStore` (stable between notifications). */
  snapshot: ClockSnapshot

  private clamp(t: number) {
    return Math.min(this.max, Math.max(this.min, t))
  }

  time(): number {
    if (!this.playing) return this.base
    return this.clamp(this.base + ((this.now() - this.anchor) / 1000) * this.speed)
  }

  play() {
    if (this.time() >= this.max) this.base = this.min
    this.anchor = this.now()
    this.playing = true
    this.emit()
  }

  pause() {
    this.base = this.time()
    this.playing = false
    this.emit()
  }

  toggle() {
    if (this.playing) this.pause()
    else this.play()
  }

  seek(t: number) {
    this.base = this.clamp(t)
    this.anchor = this.now()
    this.emit()
  }

  setSpeed(speed: number) {
    this.base = this.time()
    this.anchor = this.now()
    this.speed = speed
    this.emit()
  }

  /** Called every animation frame: stops at the end and notifies UI subscribers at ~10 Hz while playing. */
  tick() {
    if (!this.playing) return
    if (this.time() >= this.max) return this.pause()
    if (this.now() - this.lastNotify >= NOTIFY_MS) this.emit()
  }

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn)
    return () => void this.listeners.delete(fn)
  }

  getSnapshot = (): ClockSnapshot => this.snapshot

  private emit() {
    this.lastNotify = this.now()
    this.snapshot = { t: this.time(), playing: this.playing, speed: this.speed }
    for (const fn of this.listeners) fn()
  }
}

export const SPEEDS = [0.25, 0.5, 1, 2, 4] as const

/** Next / previous preset speed. */
export function shiftSpeed(speed: number, dir: 1 | -1): number {
  const i = SPEEDS.findIndex((s) => s >= speed)
  return SPEEDS[Math.min(SPEEDS.length - 1, Math.max(0, (i < 0 ? SPEEDS.length - 1 : i) + dir))]
}

/** Pause and move by `seconds` (± one frame for frame stepping). */
export function stepFrame(clock: MasterClock, seconds: number) {
  clock.pause()
  clock.seek(clock.time() + seconds)
}

/** Position in the camera's (proxy) video for a common-clock time. */
export function videoTime(ts: number, frameOffset: number, fps: number): number {
  return ts - frameOffset / fps
}

export interface VideoState {
  currentTime: number
  duration: number
  seeking: boolean
}

export interface SyncAction {
  /** Hard seek target (video seconds) or null. */
  seekTo: number | null
  playbackRate: number
  /** False when the camera has no video at this moment (starts later / already ended). */
  inRange: boolean
}

/** Playback-rate change per second of drift (relative); time constant ≈ 1 / (GAIN · speed) wall seconds. */
const GAIN = 3
const MAX_NUDGE = 0.5
const HARD_SEEK_S = 0.5

/**
 * Drift correction for one video: while playing, drift beyond half a frame is removed by nudging the playback
 * rate (no visible jump); drift beyond HARD_SEEK_S (scaled by speed) or any drift while paused → seek.
 */
export function syncVideo(target: number, v: VideoState, fps: number, playing: boolean, speed: number): SyncAction {
  const frame = 1 / fps
  const inRange = target >= 0 && (!Number.isFinite(v.duration) || target <= v.duration)
  if (!inRange || v.seeking) return { seekTo: null, playbackRate: speed, inRange }
  const drift = v.currentTime - target // > 0: video ahead
  const abs = Math.abs(drift)
  if (!playing) return { seekTo: abs > frame / 2 ? target : null, playbackRate: speed, inRange }
  if (abs > Math.max(HARD_SEEK_S, 2 * frame) * Math.max(1, speed)) return { seekTo: target, playbackRate: speed, inRange }
  if (abs <= frame / 2) return { seekTo: null, playbackRate: speed, inRange }
  const nudge = Math.min(MAX_NUDGE, Math.max(-MAX_NUDGE, -GAIN * drift))
  return { seekTo: null, playbackRate: speed * (1 + nudge), inRange }
}
