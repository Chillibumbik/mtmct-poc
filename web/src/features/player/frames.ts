/** Box lookup for the overlay: windows of `/runs/{id}/tracks` → per-camera time index (pred and GT layers). */
import type { Box, TrackBoxes } from '@/api/types.gen'

export interface IndexedFrame {
  ts: number
  pred: Box[]
  gt: Box[]
  /** GT person matched to each `pred` box (IoU ≥ 0.5), null if none. */
  match: (number | null)[]
}

export interface Layer {
  ts: number[]
  frames: IndexedFrame[]
  /** Max distance (s) from a stored frame at which its boxes are still shown. */
  tol: number
}

export interface CameraMeta {
  fps: number
  frame_offset: number
  width: number
  height: number
}

export interface CameraIndex {
  meta: CameraMeta
  pred: Layer
  gt: Layer
  /** (GT person, global id) per matched prediction box — input for the majority mapping. */
  votes: [number, number][]
}

/** Index of the element of sorted `ts` closest to `t`; -1 if empty. */
export function nearestIndex(ts: ArrayLike<number>, t: number): number {
  const n = ts.length
  if (n === 0) return -1
  let lo = 0
  let hi = n - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (ts[mid] < t) lo = mid + 1
    else hi = mid
  }
  // lo = first index with ts >= t (or n-1)
  return lo > 0 && t - ts[lo - 1] <= ts[lo] - t ? lo - 1 : lo
}

export function frameAt(layer: Layer, t: number): IndexedFrame | null {
  const i = nearestIndex(layer.ts, t)
  return i >= 0 && Math.abs(layer.ts[i] - t) <= layer.tol + 1e-9 ? layer.frames[i] : null
}

function iou(a: Box, b: Box): number {
  const iw = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)
  const ih = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)
  if (iw <= 0 || ih <= 0) return 0
  const inter = iw * ih
  return inter / (a.w * a.h + b.w * b.h - inter)
}

/** Greedy one-to-one matching by IoU (highest first); returns the GT id per prediction or null. */
export function matchFrame(pred: Box[], gt: Box[], thr = 0.5): (number | null)[] {
  const out: (number | null)[] = pred.map(() => null)
  if (!pred.length || !gt.length) return out
  const pairs: [number, number, number][] = []
  pred.forEach((p, i) =>
    gt.forEach((g, j) => {
      const v = iou(p, g)
      if (v >= thr) pairs.push([v, i, j])
    }),
  )
  pairs.sort((a, b) => b[0] - a[0])
  const usedGt = new Set<number>()
  for (const [, i, j] of pairs) {
    if (out[i] !== null || usedGt.has(j)) continue
    out[i] = gt[j].id
    usedGt.add(j)
  }
  return out
}

function layer(frames: IndexedFrame[], fps: number): Layer {
  const ts = frames.map((f) => f.ts)
  const gaps = ts
    .slice(1)
    .map((t, i) => t - ts[i])
    .sort((a, b) => a - b)
  const median = gaps.length ? gaps[gaps.length >> 1] : 0
  return { ts, frames, tol: Math.max(median, 1 / fps) }
}

/** All loaded windows of one camera (any order) → time index. */
export function buildCameraIndex(windows: TrackBoxes[]): CameraIndex {
  const w0 = windows[0]
  const meta: CameraMeta = { fps: w0.fps, frame_offset: w0.frame_offset, width: w0.width, height: w0.height }
  const frames: IndexedFrame[] = windows
    .flatMap((w) => w.frames)
    .sort((a, b) => a.ts - b.ts)
    .map((f) => ({ ts: f.ts, pred: f.pred, gt: f.gt, match: matchFrame(f.pred, f.gt) }))
  const votes: [number, number][] = []
  for (const f of frames) f.match.forEach((g, i) => g !== null && votes.push([g, f.pred[i].id]))
  return {
    meta,
    pred: layer(
      frames.filter((f) => f.pred.length > 0),
      meta.fps,
    ),
    gt: layer(
      frames.filter((f) => f.gt.length > 0),
      meta.fps,
    ),
    votes,
  }
}

/** GT person → global id it is assigned most often (box-frames over all cameras); ties → smaller id. */
export function majorityMapping(votes: Iterable<[number, number]>): Map<number, number> {
  const counts = new Map<number, Map<number, number>>()
  for (const [gt, gid] of votes) {
    const c = counts.get(gt) ?? new Map<number, number>()
    c.set(gid, (c.get(gid) ?? 0) + 1)
    counts.set(gt, c)
  }
  const out = new Map<number, number>()
  for (const [gt, c] of counts) {
    let best = -1
    let bestN = -1
    for (const [gid, n] of c) if (n > bestN || (n === bestN && gid < best)) [best, bestN] = [gid, n]
    out.set(gt, best)
  }
  return out
}

export type DiffStatus = { kind: 'ok'; gt: number } | { kind: 'mismatch'; gt: number; expected: number } | { kind: 'unmatched' }

export function diffStatus(gid: number, gt: number | null, majority: Map<number, number>): DiffStatus {
  if (gt === null) return { kind: 'unmatched' }
  const expected = majority.get(gt)
  return expected === undefined || expected === gid ? { kind: 'ok', gt } : { kind: 'mismatch', gt, expected }
}
