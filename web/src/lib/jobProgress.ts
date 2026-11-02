import type { JobEvent, JobOut } from '@/api/types.gen'

/** Folds the SSE stream of `/api/jobs/{id}/events` into a progress view. */

export type StageState = 'pending' | 'running' | 'done' | 'failed' | 'skipped'
export type JobStatus = JobOut['status']

export interface StageProgress {
  name: string
  state: StageState
  progress: number
}

export interface JobProgressState {
  lastId: number
  progress: number
  stage: string | null
  etaS: number | null
  status: JobStatus | null
  order: string[]
  stages: Record<string, StageProgress>
  logs: JobEvent[]
}

export const MAX_LOG_LINES = 500

/** Stage names emitted by `mtmct.pipeline.run_pipeline`; `oracle` replaces detect+track in oracle mode. */
export const PIPELINE_STAGES = ['detect', 'track', 'oracle', 'reid', 'refine', 'portrait', 'associate', 'postprocess', 'eval'] as const

const TERMINAL: JobStatus[] = ['done', 'failed', 'cancelled']
export const isTerminal = (s: JobStatus | null | undefined) => !!s && TERMINAL.includes(s)

export const initialJobProgress = (): JobProgressState => ({
  lastId: 0,
  progress: 0,
  stage: null,
  etaS: null,
  status: null,
  order: [],
  stages: {},
  logs: [],
})

const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

export function applyJobEvent(state: JobProgressState, ev: JobEvent): JobProgressState {
  if (ev.id <= state.lastId) return state
  const stages = { ...state.stages }
  let order = state.order
  let stage = state.stage

  if (ev.stage && ev.stage !== stage) {
    if (stage && stages[stage]?.state === 'running') stages[stage] = { ...stages[stage], state: 'done', progress: 1 }
    if (!stages[ev.stage]) order = [...order, ev.stage]
    stage = ev.stage
  }
  if (ev.stage) {
    const prev = stages[ev.stage] ?? { name: ev.stage, state: 'running', progress: 0 }
    stages[ev.stage] = {
      ...prev,
      state: 'running',
      progress: ev.stage_progress == null ? prev.progress : clamp01(ev.stage_progress),
    }
  }

  const rawStatus = ev.data?.status
  const status = typeof rawStatus === 'string' && isTerminal(rawStatus as JobStatus) ? (rawStatus as JobStatus) : state.status
  if (status && status !== state.status && stage && stages[stage]) {
    stages[stage] = status === 'done' ? { ...stages[stage], state: 'done', progress: 1 } : { ...stages[stage], state: 'failed' }
  }

  const eta = ev.data?.eta_s
  const logs = state.logs.length >= MAX_LOG_LINES ? [...state.logs.slice(1 - MAX_LOG_LINES), ev] : [...state.logs, ev]
  return {
    lastId: ev.id,
    progress: ev.progress == null ? state.progress : Math.max(state.progress, clamp01(ev.progress)),
    stage,
    etaS: status === 'done' ? 0 : typeof eta === 'number' ? eta : state.etaS,
    status,
    order,
    stages,
    logs,
  }
}

/** The stream's final `end` message carries only the terminal status. */
export function applyJobEnd(state: JobProgressState, status: JobStatus): JobProgressState {
  const stages = { ...state.stages }
  const cur = state.stage ? stages[state.stage] : undefined
  if (cur && cur.state === 'running')
    stages[cur.name] = status === 'done' ? { ...cur, state: 'done', progress: 1 } : { ...cur, state: 'failed' }
  return {
    ...state,
    status,
    stages,
    progress: status === 'done' ? 1 : state.progress,
    etaS: status === 'done' ? 0 : null,
  }
}

/** Stages to render: the canonical pipeline for run jobs, otherwise stages in order of appearance. */
export function stageList(state: JobProgressState, kind: JobOut['kind']): StageProgress[] {
  if (kind !== 'run' && kind !== 'eval_only') return state.order.map((n) => state.stages[n])
  const oracle = 'oracle' in state.stages
  const canonical: string[] = PIPELINE_STAGES.filter((s) => (oracle ? s !== 'detect' && s !== 'track' : s !== 'oracle'))
  const extra = state.order.filter((n) => !canonical.includes(n))
  const all = [...canonical, ...extra]
  const current = state.stage ? all.indexOf(state.stage) : -1
  return all.map((name, i) => state.stages[name] ?? { name, state: i < current || state.status === 'done' ? 'skipped' : 'pending', progress: 0 })
}
