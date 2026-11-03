import type { Box, SequenceOut, TrackBoxes } from '@/api/types.gen'
import { colorForId } from './colors'
import { buildCameraIndex, majorityMapping } from './frames'
import { DIFF_COLORS, hitTest, overlayBoxes } from './overlay'
import { WINDOW_S, runTimeRange, windowsAt } from './windows'

const box = (id: number, x: number, w = 10): Box => ({ x, y: 0, w, h: 20, id, tracklet_id: null })
const cam = (frames: TrackBoxes['frames']): TrackBoxes => ({ camera_id: 'c', fps: 10, frame_offset: 0, width: 100, height: 100, frames })

describe('overlay modes', () => {
  // GT person 100 is global id 1 in camera A (2 frames) but 9 in camera B; person 200 is missed; id 5 has no GT
  const a = buildCameraIndex([cam([0, 1].map((f) => ({ frame_id: f, ts: f / 10, pred: [box(1, 0)], gt: [box(100, 0)] })))])
  const b = buildCameraIndex([cam([{ frame_id: 0, ts: 0, pred: [box(9, 0), box(5, 60)], gt: [box(100, 0), box(200, 30)] }])])
  const majority = majorityMapping([...a.votes, ...b.votes])

  it('pred: colour and label by global id', () => {
    const [x] = overlayBoxes(a, 0, 'pred', majority)
    expect(x).toMatchObject({ label: '1', color: colorForId(1), gid: 1 })
  })

  it('gt: GT label, coloured like the majority global id', () => {
    const boxes = overlayBoxes(b, 0, 'gt', majority)
    expect(boxes.map((x) => x.label)).toEqual(['GT 100', 'GT 200'])
    expect(boxes[0]).toMatchObject({ color: colorForId(1), gid: 1 })
    expect(boxes[1].gid).toBeNull()
  })

  it('diff: mismatch, unmatched prediction and missed GT are told apart', () => {
    const boxes = overlayBoxes(b, 0, 'diff', majority)
    const by = Object.fromEntries(boxes.map((x) => [x.label, x]))
    expect(by['9 ≠ 1 (GT 100)']).toMatchObject({ color: DIFF_COLORS.error, weight: 3 })
    expect(by['5 · нет GT']).toMatchObject({ line: 'dashed' })
    expect(by['GT 200: пропуск']).toMatchObject({ line: 'dotted' })
    expect(overlayBoxes(a, 0, 'diff', majority)[0]).toMatchObject({ label: '1', color: DIFF_COLORS.ok })
  })

  it('hit test picks the smallest box under the cursor', () => {
    const boxes = overlayBoxes(buildCameraIndex([cam([{ frame_id: 0, ts: 0, pred: [box(1, 0, 50), box(2, 10, 10)], gt: [] }])]), 0, 'pred', majority)
    expect(hitTest(boxes, 15, 5)?.gid).toBe(2)
    expect(hitTest(boxes, 40, 5)?.gid).toBe(1)
    expect(hitTest(boxes, 80, 5)).toBeNull()
  })
})

describe('time ranges', () => {
  const seq = { splits: { all: [0, 1], test: [0.5, 1] }, duration_s: 200 } as unknown as SequenceOut

  it('follows the pipeline time_range rule', () => {
    expect(runTimeRange({ data: { max_seconds: 30, t_start: 0 } }, seq, 'all')).toEqual([0, 30])
    expect(runTimeRange({ data: { max_seconds: null, t_start: 5 } }, seq, 'test')).toEqual([105, 200])
    expect(runTimeRange(null, seq, 'all')).toEqual([0, 200])
  })

  it('prefetches the next window(s) ahead of the playhead, inside the run range', () => {
    expect(windowsAt(3, 1, [0, 30])).toEqual([0, 1])
    expect(windowsAt(3, 4, [0, 30])).toEqual([0, 1, 2])
    expect(windowsAt(29, 4, [0, 30])).toEqual([3]) // 3·8=24 ≤ 29 < 32; windows past the end are not fetched
    expect(windowsAt(WINDOW_S * 13 + 1, 1, [105, 200])).toEqual([13, 14])
  })
})
