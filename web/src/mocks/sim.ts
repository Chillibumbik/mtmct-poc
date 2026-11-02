/** In-memory mock backend: runs, jobs and a time-driven job simulator emitting progress events. */
import type {
  DatasetOut,
  ErrorCase,
  ErrorsOut,
  JobEvent,
  JobOut,
  KeyMetrics,
  RunCreate,
  RunOut,
  SweepCreate,
  SweepOut,
} from '@/api/types.gen'
import { DATASETS, PROFILES, SEQUENCES_AFTER_FETCH, seq } from './fixtures'

type Json = Record<string, unknown>

function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const r3 = (v: number) => Math.round(v * 1000) / 1000
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v))

// ---------------- metrics ----------------

export interface MockRunData {
  metrics: Json
  runtime: Json
}

/** Deterministic, internally consistent metrics.json + runtime.json for a run of quality `q` (0..1). */
export function makeRunData(run: RunOut, cameras: string[], q: number): MockRunData {
  const rand = rng(run.id * 7919)
  const n = (base: number, spread = 0.04) => r3(clamp(base + (rand() - 0.5) * 2 * spread))
  const oracle = run.mode === 'oracle'
  const detQ = oracle ? 0.97 : 0.62 + 0.15 * q
  const assQ = 0.35 + 0.5 * q

  const perCam = Object.fromEntries(
    cameras.map((c) => {
      const deta = n(detQ, 0.05)
      const assa = n(Math.min(0.95, assQ + 0.15), 0.05)
      const gt = Math.round(1500 + rand() * 2500)
      const fp = oracle ? 0 : Math.round(gt * (0.06 + rand() * 0.1))
      const fn = oracle ? Math.round(gt * 0.01) : Math.round(gt * (0.12 + rand() * 0.15))
      const idsw = Math.round((1 - assa) * 40 + rand() * 8)
      return [
        c,
        {
          HOTA: r3(Math.sqrt(deta * assa)),
          DetA: deta,
          AssA: assa,
          LocA: n(oracle ? 0.99 : 0.82),
          IDF1: n((deta + assa) / 2 + 0.05),
          IDP: n((deta + assa) / 2 + 0.08),
          IDR: n((deta + assa) / 2 + 0.02),
          MOTA: r3(1 - (fp + fn + idsw) / gt),
          MOTP: n(oracle ? 0.99 : 0.79),
          IDSW: idsw,
          FP: fp,
          FN: fn,
          Frag: Math.round(idsw * 1.6 + rand() * 10),
        },
      ]
    }),
  )
  const keys = Object.keys(Object.values(perCam)[0] ?? {})
  const mean = Object.fromEntries(
    keys.map((k) => {
      const vals = Object.values(perCam).map((m) => (m as Record<string, number>)[k])
      const avg = vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length)
      return [k, ['IDSW', 'FP', 'FN', 'Frag'].includes(k) ? Math.round(avg) : r3(avg)]
    }),
  )

  const crossAssA = n(assQ, 0.03)
  const crossDetA = n(detQ - 0.03, 0.02)
  const idp = n(assQ + 0.08)
  const idr = n(assQ - 0.02)
  const cross = {
    HOTA: r3(Math.sqrt(crossAssA * crossDetA)),
    AssA: crossAssA,
    IDF1: r3((2 * idp * idr) / (idp + idr)),
    IDP: idp,
    IDR: idr,
    MOTA: n(detQ - 0.15, 0.05),
  }
  const nGt = cameras.length > 5 ? 26 : 9
  const merges = Math.round((1 - q) * 6 + rand() * 2)
  const splits = Math.round((1 - q) * 9 + rand() * 3)
  const pp = n(0.55 + 0.4 * q)
  const pr = n(0.5 + 0.4 * q)
  const bp = n(pp + 0.05)
  const br = n(pr + 0.04)
  const threshold = Number(
    (run.overrides['associate.solver.params.threshold'] as number | undefined) ?? 0.45,
  )
  const curve = Array.from({ length: 15 }, (_, i) => {
    const t = r3(0.1 + 0.05 * i)
    const p = clamp(0.98 - Math.pow(t / 0.95, 2.2) * (1 - q * 0.4))
    const r = clamp(0.2 + 0.8 * (1 - Math.exp(-t * (3 + 3 * q))))
    return { threshold: t, pairwise_p: r3(p), pairwise_r: r3(r), IDF1: r3(cross.IDF1 - Math.pow((t - 0.45) * 1.6, 2)) }
  })

  const metrics = {
    run_id: String(run.id),
    dataset: run.dataset,
    sequence: run.sequence,
    split: run.split,
    mode: run.mode,
    per_camera: perCam,
    per_camera_mean: mean,
    cross_camera: cross,
    identity: {
      pairwise_p: pp,
      pairwise_r: pr,
      pairwise_f1: r3((2 * pp * pr) / (pp + pr)),
      bcubed_p: bp,
      bcubed_r: br,
      bcubed_f: r3((2 * bp * br) / (bp + br)),
      purity: n(bp + 0.03),
      inverse_purity: n(br + 0.02),
      n_pred_ids: nGt + splits - merges,
      n_gt_ids: nGt,
      false_merges: merges,
      false_splits: splits,
      handover_success: n(0.45 + 0.45 * q),
    },
    reid: { rank1: n(0.6 + 0.3 * q), rank5: n(0.8 + 0.15 * q), mAP: n(0.45 + 0.35 * q) },
    threshold_curve: curve,
    threshold,
  }

  const frames = cameras.length * 2300
  const tiny = run.overrides['detect.name'] === 'yolox_tiny'
  const light = run.overrides['reid.name'] === 'osnet_x0_25'
  const stageS: Record<string, number> = oracle
    ? { oracle: 2, reid: (light ? 14 : 38) + rand() * 5, portrait: 2, associate: 5 + rand() * 2, postprocess: 1, eval: 9 }
    : {
        detect: (tiny ? 210 : 520) + rand() * 60,
        track: 40 + rand() * 8,
        reid: (light ? 70 : 190) + rand() * 20,
        refine: 18 + rand() * 4,
        portrait: 5,
        associate: 9 + rand() * 3,
        postprocess: 2,
        eval: 11 + rand() * 2,
      }
  const wall = Object.values(stageS).reduce((a, b) => a + b, 0)
  // Same shape as mtmct.runtime.profiling.runtime_report.
  const lat = (k: string) =>
    k === 'detect' ? [tiny ? 28 : 61, tiny ? 39 : 88] : k === 'reid' ? [light ? 1.1 : 3.4, light ? 1.9 : 5.2] : [null, null]
  const runtime = {
    stages: Object.fromEntries(
      Object.entries(stageS).map(([k, s]) => [
        k,
        { wall_s: r3(s), items: frames, fps: r3(frames / s), p50_ms: lat(k)[0], p95_ms: lat(k)[1], cached: false },
      ]),
    ),
    wall_time_s: r3(wall),
    frames_processed: frames,
    fps: r3(frames / wall),
    peak_rss_gb: r3((oracle ? 1.4 : tiny ? 2.6 : 3.4) + rand() * 0.8 + cameras.length * 0.05),
    mean_cpu_pct: r3(70 + rand() * 25),
    cache_bytes: Math.round((oracle ? 40 : 600 + rand() * 300) * 2 ** 20),
  }
  return { metrics, runtime }
}

function keyMetrics({ metrics: m, runtime: rt }: MockRunData): KeyMetrics {
  const cross = m.cross_camera as Record<string, number>
  const mean = m.per_camera_mean as Record<string, number>
  const id = m.identity as Record<string, number>
  return {
    cross_hota: cross.HOTA,
    cross_idf1: cross.IDF1,
    cross_assa: cross.AssA,
    per_camera_hota: mean.HOTA,
    per_camera_idf1: mean.IDF1,
    pairwise_f1: id.pairwise_f1,
    bcubed_f: id.bcubed_f,
    false_merges: id.false_merges,
    false_splits: id.false_splits,
    handover_success: id.handover_success,
    reid_rank1: (m.reid as Record<string, number>).rank1,
    fps: rt.fps as number,
    peak_rss_gb: rt.peak_rss_gb as number,
    wall_time_s: rt.wall_time_s as number,
  }
}

const EMPTY_KEYS: KeyMetrics = {
  cross_hota: null,
  cross_idf1: null,
  cross_assa: null,
  per_camera_hota: null,
  per_camera_idf1: null,
  pairwise_f1: null,
  bcubed_f: null,
  false_merges: null,
  false_splits: null,
  handover_success: null,
  reid_rank1: null,
  fps: null,
  peak_rss_gb: null,
  wall_time_s: null,
}

// ---------------- state ----------------

interface SimJob extends JobOut {
  durationMs: number
  startedMs: number | null
  plan: PlanStep[]
  emitted: number
  events: JobEvent[]
}

interface PlanStep {
  at: number
  stage: string | null
  stage_progress: number | null
  level: JobEvent['level']
  message: string
}

const now = () => Date.now()
const iso = (ms: number) => new Date(ms).toISOString()

export const db = {
  datasets: structuredClone(DATASETS) as DatasetOut[],
  runs: [] as RunOut[],
  runData: new Map<number, MockRunData>(),
  quality: new Map<number, number>(),
  jobs: [] as SimJob[],
  sweeps: [] as SweepOut[],
  eventId: 0,
}

// Stage names and weights as in mtmct.pipeline.run.
const RUN_STAGES: [string, number][] = [
  ['detect', 0.4],
  ['track', 0.07],
  ['oracle', 0.05],
  ['reid', 0.27],
  ['refine', 0.05],
  ['portrait', 0.03],
  ['associate', 0.07],
  ['postprocess', 0.03],
  ['eval', 0.06],
]

function runPlan(run: RunOut, cameras: string[]): PlanStep[] {
  const skip = run.mode === 'full' ? ['oracle'] : ['detect', 'track', 'refine']
  const stages = RUN_STAGES.filter(([s]) => !skip.includes(s))
  const total = stages.reduce((a, [, w]) => a + w, 0)
  const steps: PlanStep[] = [{ at: 0, stage: null, stage_progress: null, level: 'info', message: `Старт прогона #${run.id}: ${run.dataset}/${run.sequence} (${run.split}, ${run.mode})` }]
  let acc = 0
  for (const [stage, w] of stages) {
    const span = w / total
    const ticks = stage === 'detect' || stage === 'reid' ? 8 : 3
    for (let i = 1; i <= ticks; i++) {
      const sp = i / ticks
      const cam = cameras[Math.min(cameras.length - 1, Math.floor(sp * cameras.length - 1e-9))]
      const message =
        stage === 'detect'
          ? `${cam}: кадр ${Math.round(sp * 2300)}/2300 · ${(9 + Math.sin(i) * 1.5).toFixed(1)} fps`
          : stage === 'reid'
            ? `${cam}: эмбеддинги ${Math.round(sp * 5400)}/5400 рамок`
            : stage === 'associate'
              ? i === ticks
                ? `решатель ${String(run.overrides['associate.solver.name'] ?? 'agglomerative')}: 412 треклетов → 14 глобальных ID`
                : `матрица стоимостей ${i}/${ticks - 1}`
              : `${stage}: ${Math.round(sp * 100)}%`
      steps.push({ at: acc + span * sp, stage, stage_progress: sp, level: 'info', message })
    }
    if (stage === 'associate' && run.profile === 'nonoverlap')
      steps.push({ at: acc + span, stage, stage_progress: 1, level: 'warning', message: 'Нет калибровки: стоимость geometry отключена' })
    if (stage === 'detect' && run.id % 2 === 0)
      steps.push({ at: acc + span * 0.5, stage, stage_progress: 0.5, level: 'debug', message: 'кеш detect-5f1c0a2e не найден, считаем' })
    acc += span
  }
  return steps
}

function fetchPlan(ds: DatasetOut): PlanStep[] {
  const gb = ds.download_gb
  const steps: PlanStep[] = []
  for (let i = 1; i <= 8; i++)
    steps.push({ at: 0.6 * (i / 8), stage: 'download', stage_progress: i / 8, level: 'info', message: `скачано ${((gb * i) / 8).toFixed(2)} / ${gb} ГБ` })
  steps.push({ at: 0.68, stage: 'extract', stage_progress: 1, level: 'info', message: 'распаковка архива' })
  steps.push({ at: 0.8, stage: 'ingest', stage_progress: 1, level: 'info', message: 'приведение к каноническому формату' })
  for (let i = 1; i <= 3; i++)
    steps.push({ at: 0.8 + 0.2 * (i / 3), stage: 'proxy', stage_progress: i / 3, level: 'info', message: `прокси-видео H.264 ${i}/3` })
  return steps
}

function newJob(kind: JobOut['kind'], title: string, durationMs: number, plan: PlanStep[], extra: Partial<JobOut>): SimJob {
  const id = (db.jobs.at(-1)?.id ?? 100) + 1
  const job: SimJob = {
    id,
    kind,
    status: 'queued',
    title,
    created_at: iso(now()),
    started_at: null,
    finished_at: null,
    progress: 0,
    stage: null,
    stage_progress: null,
    eta_s: null,
    error: null,
    run_id: null,
    dataset_id: null,
    ...extra,
    durationMs,
    startedMs: null,
    plan,
    emitted: 0,
    events: [],
  }
  db.jobs.push(job)
  return job
}

function emit(job: SimJob, p: Partial<JobEvent> & { message: string }) {
  const ev: JobEvent = {
    id: ++db.eventId,
    job_id: job.id,
    ts: iso(now()),
    level: 'info',
    stage: job.stage,
    progress: job.progress,
    stage_progress: job.stage_progress,
    data: {},
    ...p,
  }
  job.events.push(ev)
}

function finishJob(job: SimJob, status: 'done' | 'failed' | 'cancelled', error: string | null = null) {
  job.status = status
  job.finished_at = iso(now())
  job.error = error
  job.eta_s = status === 'done' ? 0 : null
  if (status === 'done') job.progress = 1
  emit(job, { level: status === 'done' ? 'info' : 'error', message: status === 'done' ? 'Готово' : (error ?? 'Отменено'), data: { status } })
  if (job.run_id != null) {
    const run = db.runs.find((r) => r.id === job.run_id)
    if (run) {
      run.status = status
      run.finished_at = job.finished_at
      run.error = error
      if (status === 'done') {
        const data = makeRunData(run, camerasOf(run), db.quality.get(run.id) ?? 0.6)
        db.runData.set(run.id, data)
        run.key_metrics = keyMetrics(data)
      }
    }
  }
  if (job.dataset_id) {
    const ds = db.datasets.find((d) => d.id === job.dataset_id)
    if (ds) {
      ds.job_id = null
      ds.status = status === 'done' ? 'ready' : status === 'failed' ? 'error' : 'absent'
      ds.status_message = error ?? ''
      if (status === 'done' && ds.sequences.length === 0) ds.sequences = SEQUENCES_AFTER_FETCH[ds.id] ?? []
    }
  }
}

/** Advances the simulation to the current time (called by every handler). */
export function tick() {
  const t = now()
  for (const job of db.jobs) {
    if (job.status !== 'running' || job.startedMs === null) continue
    const f = (t - job.startedMs) / job.durationMs
    while (job.emitted < job.plan.length && job.plan[job.emitted].at <= f) {
      const step = job.plan[job.emitted++]
      job.progress = Math.max(job.progress, Math.min(0.999, step.at))
      job.stage = step.stage ?? job.stage
      job.stage_progress = step.stage_progress
      job.eta_s = Math.max(0, Math.round(((1 - f) * job.durationMs) / 1000))
      emit(job, { level: step.level, message: step.message, stage: job.stage, data: { eta_s: job.eta_s } })
      const ds = job.dataset_id ? db.datasets.find((d) => d.id === job.dataset_id) : undefined
      if (ds) ds.status = step.stage === 'download' ? 'downloading' : 'preparing'
    }
    if (f >= 1) finishJob(job, 'done')
  }
  if (!db.jobs.some((j) => j.status === 'running')) {
    const next = db.jobs.find((j) => j.status === 'queued')
    if (next) startJob(next, t)
  }
}

function startJob(job: SimJob, t: number) {
  job.status = 'running'
  job.startedMs = t
  job.started_at = iso(t)
  if (job.run_id != null) {
    const run = db.runs.find((r) => r.id === job.run_id)
    if (run) run.status = 'running'
  }
  if (job.kind === 'sweep') finishJob(job, 'done')
}

// ---------------- commands ----------------

function camerasOf(run: RunOut): string[] {
  return db.datasets.find((d) => d.id === run.dataset)?.sequences.find((s) => s.name === run.sequence)?.cameras ?? ['c0']
}

function datasetDefaultProfile(dataset: string): 'overlap' | 'nonoverlap' {
  return db.datasets.find((d) => d.id === dataset)?.overlap === 'nonoverlap' ? 'nonoverlap' : 'overlap'
}

export function createRun(body: RunCreate, opts: { quality?: number; createdMs?: number } = {}): RunOut {
  const id = (db.runs.at(-1)?.id ?? 0) + 1
  const profile = body.profile ?? datasetDefaultProfile(body.dataset)
  const overrides = body.overrides ?? {}
  const run: RunOut = {
    id,
    name: body.name || `${body.dataset}/${body.sequence} · ${body.mode ?? 'full'}`,
    dataset: body.dataset,
    sequence: body.sequence,
    split: body.split ?? 'all',
    mode: body.mode ?? 'full',
    profile,
    status: 'queued',
    created_at: iso(opts.createdMs ?? now()),
    finished_at: null,
    job_id: null,
    sweep_id: body.parent_sweep_id ?? null,
    overrides,
    config: null,
    config_diff: { ...overrides },
    git_sha: 'ac375cd',
    key_metrics: { ...EMPTY_KEYS },
    error: null,
  }
  db.runs.push(run)
  db.quality.set(id, opts.quality ?? 0.45 + ((id * 37) % 40) / 100)
  const job = newJob('run', run.name, run.mode === 'oracle' ? 15_000 : 40_000, runPlan(run, camerasOf(run)), { run_id: id })
  if (opts.createdMs) job.created_at = run.created_at
  run.job_id = job.id
  return run
}

export function createSweep(body: SweepCreate): SweepOut {
  const keys = Object.keys(body.grid)
  const combos = keys.reduce<Json[]>((acc, k) => acc.flatMap((c) => body.grid[k].map((v) => ({ ...c, [k]: v }))), [{}])
  const id = (db.sweeps.at(-1)?.id ?? 0) + 1
  const name = body.name || `sweep #${id}: ${keys.join(', ')}`
  const job = newJob('sweep', name, 1000, [], {})
  const runIds = combos.map(
    (c, i) =>
      createRun({
        dataset: body.dataset,
        sequence: body.sequence,
        split: body.split ?? 'tune',
        mode: body.mode ?? 'full',
        profile: body.profile,
        overrides: { ...(body.base_overrides ?? {}), ...c },
        name: `${name} [${i + 1}/${combos.length}]`,
        parent_sweep_id: id,
      }).id,
  )
  const sweep: SweepOut = { id, name, created_at: iso(now()), n_runs: runIds.length, run_ids: runIds, job_id: job.id }
  db.sweeps.push(sweep)
  return sweep
}

export function fetchDataset(id: string): SimJob | null {
  const ds = db.datasets.find((d) => d.id === id)
  if (!ds) return null
  const job = newJob('dataset_fetch', `Скачать и подготовить ${ds.title}`, 25_000, fetchPlan(ds), { dataset_id: id })
  ds.job_id = job.id
  ds.status = 'downloading'
  ds.status_message = ''
  return job
}

export function uploadCustom(name: string, videos: string[]): SimJob {
  const id = `custom_${name.toLowerCase().replace(/[^a-z0-9]+/g, '_') || db.datasets.length}`
  const ds: DatasetOut = {
    id,
    title: name,
    licence: 'собственные данные',
    source_url: '',
    overlap: 'mixed',
    real: true,
    n_cameras: String(videos.length),
    download_gb: 0,
    requires_auth: false,
    description: `Загружено: ${videos.join(', ')}`,
    status: 'preparing',
    status_message: '',
    job_id: null,
    sequences: [],
    custom: true,
  }
  db.datasets.push(ds)
  SEQUENCES_AFTER_FETCH[id] = [{ ...seq(id, 'upload', videos.length, 60, { all: [0, 1] }, false), has_gt: false }]
  const job = newJob('dataset_prepare', `Подготовить ${name}`, 12_000, fetchPlan(ds).filter((s) => s.stage !== 'download'), {
    dataset_id: id,
  })
  ds.job_id = job.id
  return job
}

export function cancelJob(id: number): SimJob | null {
  const job = db.jobs.find((j) => j.id === id)
  if (!job) return null
  if (job.status === 'queued' || job.status === 'running') finishJob(job, 'cancelled', 'Отменено пользователем')
  return job
}

export function restartJob(id: number): SimJob | null {
  const old = db.jobs.find((j) => j.id === id)
  if (!old || !['failed', 'cancelled'].includes(old.status)) return null
  if (old.dataset_id) return fetchDataset(old.dataset_id)
  const run = db.runs.find((r) => r.id === old.run_id)
  if (!run) return null
  const job = newJob('run', run.name, old.durationMs, runPlan(run, camerasOf(run)), { run_id: run.id })
  Object.assign(run, { status: 'queued', error: null, finished_at: null, job_id: job.id })
  return job
}

export function publicJob(j: SimJob): JobOut {
  const { durationMs: _d, startedMs: _s, plan: _p, emitted: _e, events: _ev, ...out } = j
  return out
}

export function runDetail(run: RunOut): RunOut {
  const profile = PROFILES.find((p) => p.name === run.profile) ?? PROFILES[0]
  const config = structuredClone(profile.config) as Json
  for (const [k, v] of Object.entries(run.overrides)) {
    const parts = k.split('.')
    let node = config
    for (const p of parts.slice(0, -1)) node = (node[p] ??= {}) as Json
    node[parts.at(-1)!] = v
  }
  return { ...run, config }
}

export function runErrors(run: RunOut): ErrorsOut {
  const rand = rng(run.id * 31)
  const cams = camerasOf(run)
  const q = db.quality.get(run.id) ?? 0.6
  const transitions = cams.flatMap((a) =>
    cams
      .filter((b) => b !== a)
      .map((b) => {
        const total = Math.round(2 + rand() * 14)
        return { from_camera: a, to_camera: b, total, kept: Math.round(total * clamp(0.4 + 0.5 * q + (rand() - 0.5) * 0.3)) }
      }),
  )
  const mk = (kind: ErrorCase['kind'], i: number): ErrorCase => {
    const g = 3 + i * 2
    const c = cams[i % cams.length]
    const tracklets = [0, 1].map((k) => {
      const tid = 100 * (i + 1) + k
      const t0 = 10 + i * 17 + k * 6
      return {
        tracklet_id: tid,
        camera_id: cams[(i + k) % cams.length],
        t_start: t0,
        t_end: t0 + 5 + rand() * 10,
        crop_urls: [0, 1, 2].map((n) => `/api/runs/${run.id}/crops/${tid}?i=${n}`),
        gt_person_id: kind === 'false_merge' ? 10 + k : 10 + i,
        global_id: kind === 'false_merge' ? g : g + k,
      }
    })
    return {
      kind,
      global_ids: [...new Set(tracklets.map((t) => t.global_id))],
      gt_person_ids: [...new Set(tracklets.map((t) => t.gt_person_id))],
      tracklets,
      t: tracklets[0].t_start,
      camera_id: c,
    }
  }
  const id = (db.runData.get(run.id)?.metrics.identity ?? {}) as Record<string, number>
  return {
    false_merges: Array.from({ length: id.false_merges ?? 0 }, (_, i) => mk('false_merge', i)),
    false_splits: Array.from({ length: id.false_splits ?? 0 }, (_, i) => mk('false_split', i)),
    transitions,
  }
}

// ---------------- seed ----------------

/** Fills a historical job with its plan events (up to fraction `upTo`) spread over `spanMs`. */
function replay(job: SimJob, startMs: number, spanMs: number, upTo = 1) {
  job.status = 'running'
  job.startedMs = startMs
  job.started_at = iso(startMs)
  while (job.emitted < job.plan.length && job.plan[job.emitted].at <= upTo) {
    const s = job.plan[job.emitted++]
    job.stage = s.stage ?? job.stage
    job.progress = s.at
    job.events.push({
      id: ++db.eventId,
      job_id: job.id,
      ts: iso(startMs + s.at * spanMs),
      level: s.level,
      stage: job.stage,
      progress: s.at,
      stage_progress: s.stage_progress,
      message: s.message,
      data: {},
    })
  }
}

function seedState() {
  const t = now()
  const h = 3600_000
  const seeds: [RunCreate, number, number][] = [
    [{ dataset: 'epfl_terrace', sequence: 'terrace1', split: 'all', mode: 'oracle', name: 'terrace · oracle baseline' }, 0.85, 52],
    [{ dataset: 'epfl_terrace', sequence: 'terrace1', split: 'all', mode: 'full', name: 'terrace · full baseline' }, 0.62, 50],
    [{ dataset: 'campus', sequence: 'garden1', split: 'tune', mode: 'full', name: 'garden1 · baseline' }, 0.6, 30],
    [{ dataset: 'campus', sequence: 'garden2', split: 'test', mode: 'full', name: 'garden2 · baseline' }, 0.58, 28],
    [{ dataset: 'campus', sequence: 'garden2', split: 'test', mode: 'oracle', name: 'garden2 · oracle' }, 0.82, 27],
    [
      { dataset: 'campus', sequence: 'garden2', split: 'test', mode: 'full', name: 'garden2 · yolox_tiny + rerank', overrides: { 'detect.name': 'yolox_tiny', 'associate.rerank': { k1: 20, k2: 6, lambda_: 0.3 } } },
      0.66,
      20,
    ],
    [{ dataset: 'chirla', sequence: 'seq_004', split: 'test', mode: 'full', name: 'chirla seq_004 · baseline' }, 0.42, 8],
    [
      { dataset: 'chirla', sequence: 'seq_004', split: 'test', mode: 'full', name: 'chirla seq_004 · anchor solver', overrides: { 'associate.solver.name': 'anchor', 'associate.costs.time.weight': 0.8 } },
      0.5,
      6,
    ],
  ]
  for (const [body, q, hoursAgo] of seeds) {
    const run = createRun(body, { quality: q, createdMs: t - hoursAgo * h })
    const job = db.jobs.find((j) => j.id === run.job_id)!
    const span = run.mode === 'oracle' ? 90_000 : 14 * 60_000
    replay(job, t - hoursAgo * h, span)
    finishJob(job, 'done')
    job.finished_at = run.finished_at = iso(t - hoursAgo * h + span)
  }
  // a sweep on the tune split (threshold), finished
  const sweep = createSweep({
    dataset: 'campus',
    sequence: 'garden1',
    split: 'tune',
    mode: 'oracle',
    grid: { 'associate.solver.params.threshold': [0.35, 0.45, 0.55] },
    name: 'garden1 · threshold sweep',
  })
  const sweepJob = db.jobs.find((j) => j.id === sweep.job_id)!
  sweepJob.created_at = sweep.created_at = iso(t - 24 * h)
  finishJob(sweepJob, 'done')
  sweep.run_ids.forEach((rid, i) => {
    db.quality.set(rid, [0.7, 0.8, 0.68][i])
    const run = db.runs.find((r) => r.id === rid)!
    const job = db.jobs.find((j) => j.run_id === rid)!
    run.created_at = job.created_at = iso(t - 24 * h)
    replay(job, t - 24 * h + i * 100_000, 90_000)
    finishJob(job, 'done')
    job.finished_at = run.finished_at = iso(t - 24 * h + (i + 1) * 100_000)
  })
  // failed run (RAM guard)
  const failed = createRun({ dataset: 'chirla', sequence: 'seq_001', split: 'tune', mode: 'full', name: 'chirla seq_001 · omz_0287', overrides: { 'reid.name': 'omz_0287', 'reid.params.batch': 128 } }, { createdMs: t - 4 * h })
  const fj = db.jobs.find((j) => j.id === failed.job_id)!
  replay(fj, t - 4 * h, 14 * 60_000, 0.58)
  finishJob(fj, 'failed', 'RAM guard: пиковый RSS 9.21 ГБ > data.max_rss_gb = 9.0')

  // running run (started 35% ago) and a queued one
  const running = createRun({ dataset: 'campus', sequence: 'garden2', split: 'test', mode: 'full', name: 'garden2 · osnet_x0_25', overrides: { 'reid.name': 'osnet_x0_25' } }, { quality: 0.55 })
  const rj = db.jobs.find((j) => j.id === running.job_id)!
  startJob(rj, t - rj.durationMs * 0.35)
  createRun({ dataset: 'chirla', sequence: 'seq_001', split: 'tune', mode: 'oracle', name: 'chirla seq_001 · oracle' })
}

seedState()
tick()
