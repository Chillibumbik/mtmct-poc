/** TanStack Query hooks for every endpoint of spec §10.1. */
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useReducer } from 'react'
import { applyJobEnd, applyJobEvent, initialJobProgress, isTerminal, type JobProgressState } from '@/lib/jobProgress'
import { api, uploadCustomDataset, urls, type CustomUpload, type RunFilters, type TrackQuery } from './client'
import type { DatasetOut, JobEvent, JobOut, RunCreate, SweepCreate } from './types.gen'

export const qk = {
  system: ['system'] as const,
  datasets: ['datasets'] as const,
  registry: ['registry'] as const,
  profiles: ['profiles'] as const,
  runs: (f: RunFilters = {}) => ['runs', f] as const,
  run: (id: number) => ['run', id] as const,
  runMetrics: (id: number) => ['run', id, 'metrics'] as const,
  runErrors: (id: number) => ['run', id, 'errors'] as const,
  runTracks: (id: number, q: TrackQuery) => ['run', id, 'tracks', q] as const,
  runTrajectories: (id: number) => ['run', id, 'trajectories'] as const,
  person: (id: number, gid: number) => ['run', id, 'person', gid] as const,
  compare: (ids: number[]) => ['compare', ids] as const,
  jobs: ['jobs'] as const,
  job: (id: number) => ['job', id] as const,
}

const ACTIVE_JOB: JobOut['status'][] = ['queued', 'running']
const ACTIVE_DATASET: DatasetOut['status'][] = ['downloading', 'preparing']
const POLL_MS = 2000

export const useSystem = () => useQuery({ queryKey: qk.system, queryFn: api.system, staleTime: 30_000 })

export const useDatasets = () =>
  useQuery({
    queryKey: qk.datasets,
    queryFn: api.datasets,
    refetchInterval: (q) => (q.state.data?.some((d) => ACTIVE_DATASET.includes(d.status)) ? POLL_MS : false),
  })

export const useRegistry = () => useQuery({ queryKey: qk.registry, queryFn: api.registry, staleTime: Infinity })
export const useProfiles = () => useQuery({ queryKey: qk.profiles, queryFn: api.profiles, staleTime: Infinity })

export const useRuns = (f: RunFilters = {}) =>
  useQuery({
    queryKey: qk.runs(f),
    queryFn: () => api.runs(f),
    refetchInterval: (q) => (q.state.data?.some((r) => ACTIVE_JOB.includes(r.status)) ? POLL_MS * 2 : false),
  })

export const useRun = (id: number) =>
  useQuery({
    queryKey: qk.run(id),
    queryFn: () => api.run(id),
    refetchInterval: (q) => (q.state.data && ACTIVE_JOB.includes(q.state.data.status) ? POLL_MS : false),
  })

export const useRunMetrics = (id: number, enabled = true) =>
  useQuery({ queryKey: qk.runMetrics(id), queryFn: () => api.runMetrics(id), enabled, staleTime: Infinity })

/** Metrics of many runs at once (resources page); results keep the order of `ids`. */
export const useManyRunMetrics = (ids: number[]) =>
  useQueries({
    queries: ids.map((id) => ({ queryKey: qk.runMetrics(id), queryFn: () => api.runMetrics(id), staleTime: Infinity })),
  })

export const useRunErrors = (id: number, enabled = true) =>
  useQuery({ queryKey: qk.runErrors(id), queryFn: () => api.runErrors(id), enabled, staleTime: Infinity })

export const useRunTracks = (id: number, q: TrackQuery, enabled = true) =>
  useQuery({ queryKey: qk.runTracks(id, q), queryFn: () => api.runTracks(id, q), enabled, staleTime: Infinity })

export const useRunTrajectories = (id: number, enabled = true) =>
  useQuery({ queryKey: qk.runTrajectories(id), queryFn: () => api.runTrajectories(id), enabled, staleTime: Infinity })

export const usePerson = (id: number, gid: number, enabled = true) =>
  useQuery({ queryKey: qk.person(id, gid), queryFn: () => api.person(id, gid), enabled, staleTime: Infinity })

export const useCompare = (ids: number[]) =>
  useQuery({ queryKey: qk.compare(ids), queryFn: () => api.compare(ids), enabled: ids.length > 0 })

export const useJobs = () =>
  useQuery({
    queryKey: qk.jobs,
    queryFn: api.jobs,
    refetchInterval: (q) => (q.state.data?.some((j) => ACTIVE_JOB.includes(j.status)) ? POLL_MS : POLL_MS * 5),
  })

export const useJob = (id: number | null | undefined) =>
  useQuery({
    queryKey: qk.job(id ?? -1),
    queryFn: () => api.job(id!),
    enabled: id != null,
    refetchInterval: (q) => (q.state.data && ACTIVE_JOB.includes(q.state.data.status) ? POLL_MS : false),
  })

function useInvalidate() {
  const qc = useQueryClient()
  return (...keys: (readonly unknown[])[]) => Promise.all(keys.map((queryKey) => qc.invalidateQueries({ queryKey })))
}

export function useFetchDataset() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: api.fetchDataset,
    onSuccess: () => invalidate(qk.datasets, qk.jobs),
  })
}

export function useUploadCustom(onProgress?: (fraction: number) => void) {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (u: CustomUpload) => uploadCustomDataset(u, onProgress),
    onSuccess: () => invalidate(qk.datasets, qk.jobs),
  })
}

export function useCreateRun() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (body: RunCreate) => api.createRun(body),
    onSuccess: () => invalidate(['runs'], qk.jobs),
  })
}

export function useCreateSweep() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (body: SweepCreate) => api.createSweep(body),
    onSuccess: () => invalidate(['runs'], qk.jobs),
  })
}

export function useDeleteRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => api.deleteRun(id),
    onSuccess: (_, id) => {
      qc.removeQueries({ queryKey: qk.run(id) })
      return qc.invalidateQueries({ queryKey: ['runs'] })
    },
  })
}

export function useCancelJob() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (id: number) => api.cancelJob(id),
    onSuccess: (job) => invalidate(qk.jobs, qk.job(job.id), ['runs'], qk.datasets),
  })
}

export function useRestartJob() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (id: number) => api.restartJob(id),
    onSuccess: () => invalidate(qk.jobs, ['runs'], qk.datasets),
  })
}

export type StreamState = 'idle' | 'connecting' | 'open' | 'closed' | 'error'

type Action =
  | { type: 'event'; ev: JobEvent }
  | { type: 'end'; status: JobOut['status'] }
  | { type: 'reset' }
  | { type: 'conn'; conn: StreamState }

function reducer(s: JobProgressState & { conn: StreamState }, a: Action) {
  switch (a.type) {
    case 'reset':
      return { ...initialJobProgress(), conn: 'idle' as StreamState }
    case 'conn':
      return { ...s, conn: a.conn }
    case 'end':
      return { ...applyJobEnd(s, a.status), conn: s.conn }
    case 'event':
      return { ...applyJobEvent(s, a.ev), conn: s.conn }
  }
}

const parse = <T,>(text: string): T | null => {
  try {
    return JSON.parse(text) as T
  } catch {
    return null
  }
}

/**
 * Live progress of a job via SSE (`/api/jobs/{id}/events`): `job_event` messages carry a JSON `JobEvent`
 * (unnamed messages are accepted too), then one `end` message `{status}` once the job is finished.
 */
export function useJobEvents(jobId: number | null | undefined, enabled = true) {
  const [state, dispatch] = useReducer(reducer, undefined, () => ({ ...initialJobProgress(), conn: 'idle' as StreamState }))
  const qc = useQueryClient()

  useEffect(() => {
    dispatch({ type: 'reset' })
    if (jobId == null || !enabled) return
    const es = new EventSource(urls.jobEvents(jobId))
    dispatch({ type: 'conn', conn: 'connecting' })
    const finish = () => {
      es.close()
      dispatch({ type: 'conn', conn: 'closed' })
      void qc.invalidateQueries({ queryKey: qk.jobs })
      void qc.invalidateQueries({ queryKey: qk.job(jobId) })
      void qc.invalidateQueries({ queryKey: ['runs'] })
      void qc.invalidateQueries({ queryKey: qk.datasets })
    }
    es.onopen = () => dispatch({ type: 'conn', conn: 'open' })
    const onEvent = (m: MessageEvent<string>) => {
      const ev = parse<JobEvent>(m.data)
      if (ev) dispatch({ type: 'event', ev })
    }
    es.onmessage = onEvent
    es.addEventListener('job_event', onEvent)
    es.addEventListener('end', (m: MessageEvent<string>) => {
      const status = parse<{ status?: JobOut['status'] }>(m.data)?.status
      if (isTerminal(status)) dispatch({ type: 'end', status: status! })
      finish()
    })
    // EventSource reconnects by itself (CONNECTING); CLOSED means the server refused the stream.
    es.onerror = () => dispatch({ type: 'conn', conn: es.readyState === EventSource.CLOSED ? 'error' : 'connecting' })
    return () => es.close()
  }, [jobId, enabled, qc])

  return state
}
