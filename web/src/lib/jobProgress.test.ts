import { describe, expect, it } from 'vitest'
import type { JobEvent } from '@/api/types.gen'
import { MAX_LOG_LINES, applyJobEnd, applyJobEvent, initialJobProgress, stageList } from './jobProgress'

let nextId = 1
const ev = (p: Partial<JobEvent>): JobEvent => ({
  id: nextId++,
  job_id: 7,
  ts: '2026-11-03T10:00:00Z',
  level: 'info',
  stage: null,
  progress: null,
  stage_progress: null,
  message: '',
  data: {},
  ...p,
})

const fold = (events: JobEvent[]) => events.reduce(applyJobEvent, initialJobProgress())

describe('applyJobEvent', () => {
  it('tracks overall and per-stage progress', () => {
    const s = fold([
      ev({ stage: 'detect', progress: 0.1, stage_progress: 0.5, message: 'c0 100/200' }),
      ev({ stage: 'detect', progress: 0.2, stage_progress: 1 }),
    ])
    expect(s.progress).toBe(0.2)
    expect(s.stage).toBe('detect')
    expect(s.stages.detect).toMatchObject({ progress: 1, state: 'running' })
    expect(s.logs).toHaveLength(2)
  })

  it('marks previous stage done when the next one starts', () => {
    const s = fold([ev({ stage: 'detect', stage_progress: 0.7 }), ev({ stage: 'track', stage_progress: 0.1 })])
    expect(s.stages.detect).toMatchObject({ state: 'done', progress: 1 })
    expect(s.stages.track.state).toBe('running')
    expect(s.order).toEqual(['detect', 'track'])
  })

  it('ignores duplicate / replayed events (EventSource reconnect)', () => {
    const a = ev({ stage: 'detect', progress: 0.3 })
    const b = ev({ stage: 'detect', progress: 0.4 })
    const s = [a, b, a, b].reduce(applyJobEvent, initialJobProgress())
    expect(s.logs).toHaveLength(2)
    expect(s.progress).toBe(0.4)
  })

  it('never moves overall progress backwards', () => {
    const s = fold([ev({ progress: 0.5 }), ev({ progress: 0.3 })])
    expect(s.progress).toBe(0.5)
  })

  it('takes the terminal status from event data', () => {
    let s = fold([ev({ stage: 'eval', progress: 0.9 })])
    s = applyJobEvent(s, ev({ progress: 1, message: 'done', data: { status: 'done' } }))
    expect(s.status).toBe('done')
    expect(s.stages.eval.state).toBe('done')
  })

  it('marks the current stage failed on failure', () => {
    const s = fold([ev({ stage: 'reid', progress: 0.4 }), ev({ level: 'error', message: 'OOM', data: { status: 'failed' } })])
    expect(s.status).toBe('failed')
    expect(s.stages.reid.state).toBe('failed')
  })

  it('applies the terminal `end` SSE message (status only)', () => {
    const s = applyJobEnd(fold([ev({ stage: 'eval', progress: 0.95, stage_progress: 0.5 })]), 'done')
    expect(s).toMatchObject({ status: 'done', progress: 1 })
    expect(s.stages.eval.state).toBe('done')
    const c = applyJobEnd(fold([ev({ stage: 'detect', progress: 0.2 })]), 'cancelled')
    expect(c.status).toBe('cancelled')
    expect(c.stages.detect.state).toBe('failed')
    expect(c.progress).toBe(0.2)
  })

  it('caps the log buffer', () => {
    const events = Array.from({ length: MAX_LOG_LINES + 50 }, () => ev({ message: 'x' }))
    expect(fold(events).logs).toHaveLength(MAX_LOG_LINES)
  })
})

describe('stageList', () => {
  it('lists the full-mode pipeline for run jobs; unseen earlier stages are skipped', () => {
    const s = fold([ev({ stage: 'reid', stage_progress: 0.5 })])
    const list = stageList(s, 'run')
    expect(list.map((x) => x.name)).toEqual(['detect', 'track', 'reid', 'refine', 'portrait', 'associate', 'postprocess', 'eval'])
    expect(list[0]).toMatchObject({ name: 'detect', state: 'skipped' })
    expect(list.find((x) => x.name === 'reid')).toMatchObject({ state: 'running', progress: 0.5 })
    expect(list.find((x) => x.name === 'refine')!.state).toBe('pending')
  })

  it('switches to the oracle pipeline once the oracle stage is seen', () => {
    const list = stageList(fold([ev({ stage: 'oracle', stage_progress: 1 })]), 'run')
    expect(list.map((x) => x.name).slice(0, 2)).toEqual(['oracle', 'reid'])
    expect(list.some((x) => x.name === 'detect')).toBe(false)
  })
  it('lists only seen stages for other job kinds', () => {
    const s = fold([ev({ stage: 'download', stage_progress: 0.2 })])
    expect(stageList(s, 'dataset_fetch').map((x) => x.name)).toEqual(['download'])
  })
})
