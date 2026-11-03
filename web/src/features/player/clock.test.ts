import { MasterClock, shiftSpeed, syncVideo, videoTime } from './clock'

/** Simulated <video>: its own clock (optionally skewed), stalls and seek latency. */
class SimVideo {
  currentTime = 0
  duration = 1000
  seeking = false
  rate = 1
  private seekLeft = 0
  private seekTarget = 0
  private stallLeft = 0

  private readonly skew: number
  private readonly seekLatency: number

  constructor(skew = 0, seekLatency = 0.08) {
    this.skew = skew
    this.seekLatency = seekLatency
  }

  seek(t: number) {
    this.seeking = true
    this.seekTarget = t
    this.seekLeft = this.seekLatency
  }

  stall(s: number) {
    this.stallLeft = s
  }

  advance(dt: number, playing: boolean) {
    if (this.seeking) {
      this.seekLeft -= dt
      if (this.seekLeft <= 0) {
        this.seeking = false
        this.currentTime = this.seekTarget
      }
      return
    }
    if (this.stallLeft > 0) {
      this.stallLeft -= dt
      return
    }
    if (playing) this.currentTime += dt * this.rate * (1 + this.skew)
  }
}

const FPS = 25
const FRAME = 1 / FPS
const DT = 1 / 60

/** Runs master + one video for `seconds` of wall time; returns |drift| (s) per step after `from` s. */
function simulate(opts: {
  seconds: number
  speed?: number
  skew?: number
  correct?: boolean
  offsetFrames?: number
  events?: Record<number, (clock: MasterClock, video: SimVideo) => void>
}) {
  let wall = 0
  const clock = new MasterClock(0, 10_000, 0, () => wall * 1000)
  const video = new SimVideo(opts.skew ?? 0)
  const offset = opts.offsetFrames ?? 0
  clock.setSpeed(opts.speed ?? 1)
  clock.play()
  const drift: { t: number; d: number }[] = []
  const steps = Math.round(opts.seconds / DT)
  for (let i = 0; i < steps; i++) {
    opts.events?.[i]?.(clock, video)
    const target = videoTime(clock.time(), offset, FPS)
    if (opts.correct !== false) {
      const a = syncVideo(target, video, FPS, clock.playing, clock.speed)
      if (a.seekTo !== null) video.seek(a.seekTo)
      video.rate = a.playbackRate
    } else video.rate = clock.speed
    wall += DT
    video.advance(DT, clock.playing)
    drift.push({ t: wall, d: video.seeking ? NaN : video.currentTime - videoTime(clock.time(), offset, FPS) })
  }
  return { drift, clock, video }
}

const maxAbs = (xs: { t: number; d: number }[], from: number, to = Infinity) =>
  Math.max(...xs.filter((x) => x.t >= from && x.t <= to && !Number.isNaN(x.d)).map((x) => Math.abs(x.d)))

describe('test_sync_drift_correction', () => {
  it('without correction a skewed video clock drifts away (simulation sanity)', () => {
    const { drift } = simulate({ seconds: 30, skew: 0.03, correct: false })
    expect(maxAbs(drift, 29)).toBeGreaterThan(0.8)
  })

  it('keeps a fast or slow video within one frame of the master clock', () => {
    for (const skew of [0.03, -0.03])
      for (const speed of [0.25, 1, 4]) {
        const { drift } = simulate({ seconds: 40, skew, speed })
        expect(maxAbs(drift, 3), `skew ${skew} speed ${speed}`).toBeLessThanOrEqual(FRAME * 1.05)
      }
  })

  it('recovers from a playback stall by catching up, without seeking', () => {
    const stallAt = Math.round(10 / DT)
    let seeks = 0
    const { drift } = simulate({
      seconds: 20,
      events: {
        [stallAt]: (_, v) => {
          v.stall(0.3)
          const orig = v.seek.bind(v)
          v.seek = (t) => {
            seeks++
            orig(t)
          }
        },
      },
    })
    expect(maxAbs(drift, 10.2, 10.4)).toBeGreaterThan(0.2) // the stall really happened
    expect(maxAbs(drift, 12)).toBeLessThanOrEqual(FRAME * 1.05)
    expect(seeks).toBe(0)
  })

  it('hard-seeks on a large jump and then holds sync', () => {
    const { drift } = simulate({ seconds: 20, offsetFrames: 60, events: { [Math.round(5 / DT)]: (c) => c.seek(300) } })
    expect(maxAbs(drift, 6.5)).toBeLessThanOrEqual(FRAME * 1.05)
  })

  it('lands exactly on the target frame when paused', () => {
    const v = { currentTime: 3.0, duration: 100, seeking: false }
    expect(syncVideo(3.2, v, FPS, false, 1).seekTo).toBe(3.2)
    expect(syncVideo(3.01, v, FPS, false, 1).seekTo).toBeNull() // within half a frame
  })

  it('reports out-of-range times (camera starts later / ends earlier)', () => {
    const v = { currentTime: 0, duration: 10, seeking: false }
    expect(syncVideo(-0.5, v, FPS, true, 1).inRange).toBe(false)
    expect(syncVideo(10.5, v, FPS, true, 1).inRange).toBe(false)
    expect(syncVideo(5, v, FPS, true, 1).inRange).toBe(true)
    expect(videoTime(10, 50, 25)).toBeCloseTo(8)
  })
})

describe('MasterClock', () => {
  it('advances with speed, pauses, seeks and clamps to its range', () => {
    let now = 0
    const c = new MasterClock(2, 12, 2, () => now)
    expect(c.time()).toBe(2)
    c.play()
    now = 1000
    expect(c.time()).toBeCloseTo(3)
    c.setSpeed(4)
    now = 2000
    expect(c.time()).toBeCloseTo(7)
    c.pause()
    now = 9000
    expect(c.time()).toBeCloseTo(7)
    c.seek(100)
    expect(c.time()).toBe(12)
    c.play() // at the end → restarts from the beginning
    expect(c.time()).toBe(2)
  })

  it('stops at the end and notifies subscribers (throttled while playing)', () => {
    let now = 0
    const c = new MasterClock(0, 1, 0, () => now)
    const seen: number[] = []
    c.subscribe(() => seen.push(c.time()))
    c.play()
    for (now = 0; now <= 1500; now += 16) c.tick()
    expect(c.playing).toBe(false)
    expect(c.time()).toBe(1)
    expect(seen.length).toBeGreaterThan(5)
    expect(seen.length).toBeLessThan(20)
  })
})

it('steps through preset speeds 0.25–4×', () => {
  expect(shiftSpeed(1, 1)).toBe(2)
  expect(shiftSpeed(4, 1)).toBe(4)
  expect(shiftSpeed(0.25, -1)).toBe(0.25)
  expect(shiftSpeed(0.5, -1)).toBe(0.25)
})
