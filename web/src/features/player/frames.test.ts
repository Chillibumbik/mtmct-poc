import type { Box, TrackBoxes } from '@/api/types.gen'
import { buildCameraIndex, diffStatus, frameAt, majorityMapping, matchFrame, nearestIndex } from './frames'

const box = (id: number, x = 0, extra: Partial<Box> = {}): Box => ({ x, y: 0, w: 10, h: 20, id, tracklet_id: null, ...extra })

function track(frames: { f: number; pred?: Box[]; gt?: Box[] }[], fps = 25, offset = 0): TrackBoxes {
  return {
    camera_id: 'c0',
    fps,
    frame_offset: offset,
    width: 360,
    height: 288,
    frames: frames.map(({ f, pred = [], gt = [] }) => ({ frame_id: f, ts: (f + offset) / fps, pred, gt })),
  }
}

describe('test_overlay_frame_lookup', () => {
  it('finds the nearest timestamp by binary search', () => {
    const ts = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    expect(nearestIndex(ts, -1)).toBe(0)
    expect(nearestIndex(ts, 0.04)).toBe(0)
    expect(nearestIndex(ts, 0.06)).toBe(1)
    expect(nearestIndex(ts, 0.3)).toBe(3)
    expect(nearestIndex(ts, 0.41)).toBe(4)
    expect(nearestIndex(ts, 0.74)).toBe(4)
    expect(nearestIndex(ts, 0.76)).toBe(5)
    expect(nearestIndex(ts, 99)).toBe(5)
    expect(nearestIndex([], 1)).toBe(-1)
  })

  it('matches a brute-force search on random data', () => {
    let seed = 7
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    const ts = Array.from({ length: 500 }, () => rnd() * 100).sort((a, b) => a - b)
    for (let i = 0; i < 300; i++) {
      const t = rnd() * 110 - 5
      const brute = Math.min(...ts.map((v) => Math.abs(v - t)))
      expect(Math.abs(ts[nearestIndex(ts, t)] - t)).toBeCloseTo(brute, 12)
    }
  })

  it('looks frames up per layer with a tolerance of one typical gap', () => {
    // predictions processed at 10 fps of a 25 fps video (frame ids 0, 2/3, 5, ...), GT on every frame
    const predFrames = [0, 3, 5, 8, 10]
    const idx = buildCameraIndex([
      track(Array.from({ length: 11 }, (_, f) => ({ f, gt: [box(1)], pred: predFrames.includes(f) ? [box(7)] : [] }))),
    ])
    expect(idx.pred.ts).toEqual(predFrames.map((f) => f / 25))
    expect(idx.gt.ts.length).toBe(11)
    // between processed frames: nearest processed frame is shown (no flicker)
    expect(frameAt(idx.pred, 6 / 25)?.ts).toBe(5 / 25)
    expect(frameAt(idx.pred, 7 / 25)?.ts).toBe(8 / 25)
    // far past the last frame: nothing (person gone)
    expect(frameAt(idx.pred, 1)).toBeNull()
    expect(frameAt(idx.gt, 4 / 25)?.ts).toBe(4 / 25)
  })

  it('merges windows of one camera in time order and keeps camera geometry', () => {
    const a = track([{ f: 0, pred: [box(1)] }, { f: 1, pred: [box(1)] }], 25, 3)
    const b = track([{ f: 200, pred: [box(1)] }], 25, 3)
    const idx = buildCameraIndex([b, a])
    expect(idx.pred.ts).toEqual([3 / 25, 4 / 25, 203 / 25])
    expect(idx.meta).toEqual({ fps: 25, frame_offset: 3, width: 360, height: 288 })
  })
})

describe('diff mode', () => {
  it('matches predictions to GT by IoU >= 0.5', () => {
    const gt = [box(100, 0), box(200, 50)]
    expect(matchFrame([box(1, 52), box(2, 1), box(3, 500)], gt)).toEqual([200, 100, null])
    // a GT box is used once: the better-overlapping prediction wins
    expect(matchFrame([box(1, 2), box(2, 0)], [box(100, 0)])).toEqual([null, 100])
  })

  it('maps each GT person to the global id it gets most often (all cameras)', () => {
    const votes: [number, number][] = [
      [100, 1],
      [100, 1],
      [100, 9],
      [200, 2],
      [200, 3],
    ]
    const m = majorityMapping(votes)
    expect(m.get(100)).toBe(1)
    expect(m.get(200)).toBe(2) // tie → smaller id, deterministic
  })

  it('flags boxes whose id disagrees with the majority of their GT person', () => {
    const c0 = track([
      { f: 0, pred: [box(1, 0)], gt: [box(100, 0)] },
      { f: 1, pred: [box(1, 0)], gt: [box(100, 0)] },
    ])
    const c1 = track([{ f: 0, pred: [box(9, 0), box(5, 200)], gt: [box(100, 0)] }])
    const i0 = buildCameraIndex([c0])
    const i1 = buildCameraIndex([c1])
    const majority = majorityMapping([i0, i1].flatMap((i) => i.votes))
    const f1 = frameAt(i1.pred, 0)!
    expect(diffStatus(f1.pred[0].id, f1.match[0], majority)).toEqual({ kind: 'mismatch', expected: 1, gt: 100 })
    expect(diffStatus(f1.pred[1].id, f1.match[1], majority)).toEqual({ kind: 'unmatched' })
    const f0 = frameAt(i0.pred, 0)!
    expect(diffStatus(f0.pred[0].id, f0.match[0], majority)).toEqual({ kind: 'ok', gt: 100 })
  })
})
