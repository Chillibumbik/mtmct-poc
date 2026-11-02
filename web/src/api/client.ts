/** Typed fetch client for every endpoint of spec §10.1 (prefix `/api`). */
import type {
  CompareOut,
  DatasetOut,
  ErrorsOut,
  JobOut,
  PersonRoute,
  ProfileOut,
  RegistryOut,
  RunCreate,
  RunMetrics,
  RunOut,
  SweepCreate,
  SweepOut,
  SystemOut,
  TrackBoxes,
  TrajectoriesOut,
} from './types.gen'

export const API = '/api'

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

/** FastAPI errors: `{detail: string}` or `{detail: [{loc, msg}]}` (validation). */
function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail))
    return detail.map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).join('.')}: ${d.msg ?? ''}`).join('; ')
  return `HTTP ${status}`
}

function parseBody(text: string): unknown {
  try {
    return text ? JSON.parse(text) : null
  } catch {
    return text
  }
}

/** Human-readable message for any thrown value (API errors include the HTTP status). */
export function errorText(error: unknown): string {
  if (error instanceof ApiError) return error.status ? `${error.message} (HTTP ${error.status})` : error.message
  if (error instanceof Error) return error.message
  return String(error)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(API + path, {
      ...init,
      headers: init?.body && !(init.body instanceof FormData) ? { 'Content-Type': 'application/json', ...init.headers } : init?.headers,
    })
  } catch (e) {
    throw new ApiError(0, 'Сервер API недоступен', e)
  }
  if (res.status === 204) return undefined as T
  const body = parseBody(await res.text())
  if (!res.ok) throw new ApiError(res.status, errorMessage(res.status, body), body)
  return body as T
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  body: body === undefined ? undefined : JSON.stringify(body),
})

function qs(params: Record<string, string | number | undefined | null>): string {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') q.set(k, String(v))
  const s = q.toString()
  return s ? `?${s}` : ''
}

export interface RunFilters {
  dataset?: string
  sequence?: string
  split?: string
  mode?: string
  status?: string
  sweep_id?: number
  limit?: number
}

export interface TrackQuery {
  camera: string
  t0?: number
  t1?: number
}

export const api = {
  system: () => request<SystemOut>('/system'),

  datasets: () => request<DatasetOut[]>('/datasets'),
  fetchDataset: (id: string) => request<JobOut>(`/datasets/${encodeURIComponent(id)}/fetch`, json('POST')),

  registry: () => request<RegistryOut>('/registry'),
  profiles: () => request<ProfileOut[]>('/profiles'),

  createRun: (body: RunCreate) => request<RunOut>('/runs', json('POST', body)),
  createSweep: (body: SweepCreate) => request<SweepOut>('/sweeps', json('POST', body)),
  runs: (f: RunFilters = {}) => request<RunOut[]>(`/runs${qs({ ...f })}`),
  run: (id: number) => request<RunOut>(`/runs/${id}`),
  deleteRun: (id: number) => request<unknown>(`/runs/${id}`, { method: 'DELETE' }),
  runMetrics: (id: number) => request<RunMetrics>(`/runs/${id}/metrics`),
  runErrors: (id: number) => request<ErrorsOut>(`/runs/${id}/errors`),
  runTracks: (id: number, q: TrackQuery) => request<TrackBoxes>(`/runs/${id}/tracks${qs({ ...q })}`),
  runTrajectories: (id: number) => request<TrajectoriesOut>(`/runs/${id}/trajectories`),
  person: (id: number, globalId: number) => request<PersonRoute>(`/runs/${id}/person/${globalId}`),
  compare: (runIds: number[]) => request<CompareOut>(`/compare${qs({ runs: runIds.join(',') })}`),

  jobs: () => request<JobOut[]>('/jobs'),
  job: (id: number) => request<JobOut>(`/jobs/${id}`),
  cancelJob: (id: number) => request<JobOut>(`/jobs/${id}/cancel`, json('POST')),
  /** Queues a copy of a failed/cancelled job; finished stages come from the cache. */
  restartJob: (id: number) => request<JobOut>(`/jobs/${id}/restart`, json('POST')),
}

/** URLs consumed directly by the browser (img/video/EventSource/download). */
export const urls = {
  jobEvents: (jobId: number) => `${API}/jobs/${jobId}/events`,
  crops: (runId: number, trackletId: number) => `${API}/runs/${runId}/crops/${trackletId}`,
  media: (dataset: string, sequence: string, camera: string) =>
    `${API}/media/${encodeURIComponent(dataset)}/${encodeURIComponent(sequence)}/${encodeURIComponent(camera)}.mp4`,
  export: (runId: number) => `${API}/runs/${runId}/export`,
}

export interface CustomUpload {
  name: string
  videos: File[]
  calibration?: File | null
  gt?: File | null
}

/**
 * `POST /datasets/custom` (multipart: `name`, repeated `videos`, optional `calibration` JSON, optional `gt`).
 * Uses XHR for upload progress; resolves with the prepare job.
 */
export function uploadCustomDataset(u: CustomUpload, onProgress?: (fraction: number) => void): Promise<JobOut> {
  const form = new FormData()
  form.set('name', u.name)
  for (const f of u.videos) form.append('videos', f, f.name)
  if (u.calibration) form.set('calibration', u.calibration, u.calibration.name)
  if (u.gt) form.set('gt', u.gt, u.gt.name)
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API}/datasets/custom`)
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total)
    xhr.onerror = () => reject(new ApiError(0, 'Сервер API недоступен'))
    xhr.onload = () => {
      const body = parseBody(xhr.responseText)
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as JobOut)
      else reject(new ApiError(xhr.status, errorMessage(xhr.status, body), body))
    }
    xhr.send(form)
  })
}
