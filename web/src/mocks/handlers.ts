/** MSW handlers implementing spec §10.1 on top of the in-memory simulator. */
import { http, HttpResponse } from 'msw/http'
import { sse } from 'msw/sse'
import { delay } from 'msw/utils/delay'
import type { CompareOut, CompareRow, RunCreate, SweepCreate, TrackBoxes, TrajectoriesOut } from '@/api/types.gen'
import { metricDef } from '@/lib/metrics'
import { PROFILES, REGISTRY, SYSTEM, previewSvg } from './fixtures'
import { cancelJob, createRun, createSweep, db, fetchDataset, publicJob, restartJob, runDetail, runErrors, tick, uploadCustom } from './sim'

const latency = () => delay(80 + Math.random() * 160)
const notFound = (what: string) => HttpResponse.json({ detail: `${what} не найден` }, { status: 404 })
const svg = (body: string) => new HttpResponse(body, { headers: { 'Content-Type': 'image/svg+xml' } })

const runById = (id: string | readonly string[] | undefined) => db.runs.find((r) => r.id === Number(id))

function sequenceOf(dataset: string, sequence: string) {
  return db.datasets.find((d) => d.id === dataset)?.sequences.find((s) => s.name === sequence)
}

// Synthetic world for player/BEV: people walk straight lines across a 12×10 m yard.
function people(runId: number, duration: number) {
  return Array.from({ length: 8 }, (_, i) => {
    const a = ((runId + i) * 2.39996) % (2 * Math.PI)
    return {
      gid: i + 1,
      t0: (i * duration) / 12,
      t1: (i * duration) / 12 + duration / 3,
      from: [6 + 5 * Math.cos(a), 5 + 4 * Math.sin(a)] as const,
      to: [6 - 5 * Math.cos(a), 5 - 4 * Math.sin(a)] as const,
    }
  })
}

function worldAt(p: ReturnType<typeof people>[number], t: number) {
  const f = (t - p.t0) / (p.t1 - p.t0)
  return [p.from[0] + (p.to[0] - p.from[0]) * f, p.from[1] + (p.to[1] - p.from[1]) * f] as const
}

const COMPARE_KEYS: [string, string][] = [
  ['cross_camera', 'HOTA'],
  ['cross_camera', 'IDF1'],
  ['cross_camera', 'AssA'],
  ['cross_camera', 'IDP'],
  ['cross_camera', 'IDR'],
  ['identity', 'pairwise_f1'],
  ['identity', 'bcubed_f'],
  ['identity', 'false_merges'],
  ['identity', 'false_splits'],
  ['identity', 'handover_success'],
  ['reid', 'rank1'],
  ['reid', 'mAP'],
]

export const handlers = [
  http.all('/api/*', () => {
    tick()
  }),

  http.get('/api/system', async () => {
    await latency()
    return HttpResponse.json(SYSTEM)
  }),
  http.get('/api/registry', async () => {
    await latency()
    return HttpResponse.json({ components: REGISTRY })
  }),
  http.get('/api/profiles', async () => {
    await latency()
    return HttpResponse.json(PROFILES)
  }),

  http.get('/api/datasets', async () => {
    await latency()
    return HttpResponse.json(db.datasets)
  }),
  http.post('/api/datasets/:id/fetch', async ({ params }) => {
    await latency()
    const job = fetchDataset(String(params.id))
    return job ? HttpResponse.json(publicJob(job)) : notFound('Датасет')
  }),
  http.post('/api/datasets/custom', async ({ request }) => {
    const form = await request.formData()
    const name = String(form.get('name') ?? '').trim()
    const videos = form.getAll('videos').filter((v): v is File => v instanceof File)
    if (!name) return HttpResponse.json({ detail: 'Укажите название набора' }, { status: 422 })
    if (videos.length === 0) return HttpResponse.json({ detail: 'Нужен хотя бы один видеофайл' }, { status: 422 })
    const bad = videos.find((v) => !/\.(mp4|avi|mkv|mov|webm)$/i.test(v.name))
    if (bad) return HttpResponse.json({ detail: `${bad.name}: не видеофайл` }, { status: 415 })
    await delay(600)
    return HttpResponse.json(publicJob(uploadCustom(name, videos.map((v) => v.name))))
  }),
  http.get('/api/mock-preview/:ds/:seq/:cam', ({ params }) => {
    const hue = (String(params.ds).length * 47 + String(params.cam).charCodeAt(1) * 29) % 360
    return svg(previewSvg(`${params.seq} · ${params.cam}`, hue))
  }),

  http.post('/api/runs', async ({ request }) => {
    await latency()
    const body = (await request.json()) as RunCreate
    if (!sequenceOf(body.dataset, body.sequence))
      return HttpResponse.json({ detail: `Последовательность ${body.dataset}/${body.sequence} не готова` }, { status: 422 })
    return HttpResponse.json(createRun(body))
  }),
  http.post('/api/sweeps', async ({ request }) => {
    await latency()
    const body = (await request.json()) as SweepCreate
    const n = Object.values(body.grid).reduce((a, v) => a * v.length, 1)
    if (n === 0) return HttpResponse.json({ detail: 'Пустая сетка' }, { status: 422 })
    if (n > 64) return HttpResponse.json({ detail: `Слишком много прогонов: ${n} > 64` }, { status: 422 })
    return HttpResponse.json(createSweep(body))
  }),
  http.get('/api/runs', async ({ request }) => {
    await latency()
    const q = new URL(request.url).searchParams
    const runs = db.runs
      .filter((r) => !q.get('dataset') || r.dataset === q.get('dataset'))
      .filter((r) => !q.get('sequence') || r.sequence === q.get('sequence'))
      .filter((r) => !q.get('split') || r.split === q.get('split'))
      .filter((r) => !q.get('mode') || r.mode === q.get('mode'))
      .filter((r) => !q.get('status') || r.status === q.get('status'))
      .filter((r) => !q.get('sweep_id') || r.sweep_id === Number(q.get('sweep_id')))
      .slice()
      .reverse()
    return HttpResponse.json(runs)
  }),
  http.get('/api/runs/:id', async ({ params }) => {
    await latency()
    const run = runById(params.id)
    return run ? HttpResponse.json(runDetail(run)) : notFound('Прогон')
  }),
  http.delete('/api/runs/:id', async ({ params }) => {
    await latency()
    const i = db.runs.findIndex((r) => r.id === Number(params.id))
    if (i < 0) return notFound('Прогон')
    if (['queued', 'running'].includes(db.runs[i].status))
      return HttpResponse.json({ detail: 'Сначала отмените задачу прогона' }, { status: 409 })
    db.runs.splice(i, 1)
    return HttpResponse.json({ deleted: true })
  }),
  http.get('/api/runs/:id/metrics', async ({ params }) => {
    await latency()
    const data = db.runData.get(Number(params.id))
    return data ? HttpResponse.json(data) : notFound('metrics.json')
  }),
  http.get('/api/runs/:id/errors', async ({ params }) => {
    await latency()
    const run = runById(params.id)
    return run && run.status === 'done' ? HttpResponse.json(runErrors(run)) : notFound('Разбор ошибок')
  }),
  http.get('/api/runs/:id/crops/:tid', ({ params }) => svg(previewSvg(`#${params.tid}`, (Number(params.tid) * 53) % 360, 64, 128))),
  http.get('/api/runs/:id/tracks', async ({ params, request }) => {
    await latency()
    const run = runById(params.id)
    const seq = run && sequenceOf(run.dataset, run.sequence)
    if (!run || !seq) return notFound('Прогон')
    const q = new URL(request.url).searchParams
    const camera = q.get('camera') ?? seq.cameras[0]
    const ci = Math.max(0, seq.cameras.indexOf(camera))
    const t0 = Number(q.get('t0') ?? 0)
    const t1 = Math.min(Number(q.get('t1') ?? t0 + 30), t0 + 30, seq.duration_s)
    const fps = 10
    const width = 1920
    const height = 1080
    const ps = people(run.id, seq.duration_s)
    const frames: TrackBoxes['frames'] = []
    for (let f = Math.ceil(t0 * fps); f < t1 * fps; f++) {
      const t = f / fps
      const visible = ps.filter((p) => t >= p.t0 && t <= p.t1 && (p.gid + ci) % 3 !== 0)
      const boxes = visible.map((p) => {
        const [wx, wy] = worldAt(p, t)
        const h = 160 + wy * 30
        return { x: (wx / 12) * width - h * 0.2, y: height * 0.35 + wy * 40, w: h * 0.4, h, id: p.gid, tracklet_id: p.gid * 10 + ci }
      })
      frames.push({
        frame_id: f,
        ts: t,
        pred: boxes.map((b) => ({ ...b, id: b.id === 3 && ci === 1 ? 9 : b.id })),
        gt: boxes.map((b) => ({ ...b, tracklet_id: null })),
      })
    }
    return HttpResponse.json({ camera_id: camera, fps, frame_offset: 0, width, height, frames } satisfies TrackBoxes)
  }),
  http.get('/api/runs/:id/trajectories', async ({ params }) => {
    await latency()
    const run = runById(params.id)
    const seq = run && sequenceOf(run.dataset, run.sequence)
    if (!run || !seq) return notFound('Прогон')
    if (!seq.has_calibration) return HttpResponse.json({ available: false, bounds: null, trajectories: [] } satisfies TrajectoriesOut)
    const trajectories = people(run.id, seq.duration_s).map((p) => ({
      global_id: p.gid,
      points: Array.from({ length: 40 }, (_, i) => {
        const t = p.t0 + ((p.t1 - p.t0) * i) / 39
        const [x, y] = worldAt(p, t)
        return { ts: t, x, y, camera_id: seq.cameras[Math.floor((i / 40) * seq.cameras.length)] }
      }),
    }))
    return HttpResponse.json({ available: true, bounds: [0, 0, 12, 10], trajectories } satisfies TrajectoriesOut)
  }),
  http.get('/api/runs/:id/person/:gid', async ({ params }) => {
    await latency()
    const run = runById(params.id)
    const seq = run && sequenceOf(run.dataset, run.sequence)
    if (!run || !seq) return notFound('Прогон')
    const gid = Number(params.gid)
    const p = people(run.id, seq.duration_s).find((x) => x.gid === gid)
    if (!p) return notFound('Глобальный ID')
    const n = seq.cameras.length
    const segments = seq.cameras
      .filter((_, ci) => (gid + ci) % 3 !== 0)
      .map((camera_id, k) => {
        const span = (p.t1 - p.t0) / n
        const tid = gid * 10 + seq.cameras.indexOf(camera_id)
        return { camera_id, t_start: p.t0 + k * span, t_end: p.t0 + (k + 1.4) * span, tracklet_ids: [tid], crop_url: `/api/runs/${run.id}/crops/${tid}` }
      })
    return HttpResponse.json({ global_id: gid, segments, gt_person_ids: [gid] })
  }),
  http.get('/api/runs/:id/export', () => new HttpResponse('mock export', { headers: { 'Content-Type': 'application/zip' } })),

  http.get('/api/compare', async ({ request }) => {
    await latency()
    const ids = (new URL(request.url).searchParams.get('runs') ?? '').split(',').map(Number).filter(Boolean)
    const runs = ids.map((id) => runById(String(id))).filter((r) => r !== undefined)
    const rows: CompareRow[] = COMPARE_KEYS.map(([section, metric]) => {
      const values = Object.fromEntries(
        runs.map((r) => {
          const sec = db.runData.get(r.id)?.metrics[section] as Record<string, number> | undefined
          return [String(r.id), sec?.[metric] ?? null]
        }),
      )
      const first = values[String(runs[0]?.id)]
      const higher = metricDef(metric)?.higherIsBetter !== false
      const valid = runs.filter((r) => values[String(r.id)] != null)
      const best = valid.sort((a, b) => (higher ? 1 : -1) * ((values[String(b.id)] ?? 0) - (values[String(a.id)] ?? 0)))[0]
      return {
        metric: `${section}.${metric}`,
        values,
        deltas: Object.fromEntries(Object.entries(values).map(([k, v]) => [k, v == null || first == null ? null : v - first])),
        best_run_id: best?.id ?? null,
        higher_is_better: higher,
      }
    })
    return HttpResponse.json({ runs, rows } satisfies CompareOut)
  }),

  http.get('/api/jobs', async () => {
    await latency()
    return HttpResponse.json(db.jobs.map(publicJob).reverse())
  }),
  http.get('/api/jobs/:id', async ({ params }) => {
    await latency()
    const job = db.jobs.find((j) => j.id === Number(params.id))
    return job ? HttpResponse.json(publicJob(job)) : notFound('Задача')
  }),
  http.post('/api/jobs/:id/cancel', async ({ params }) => {
    await latency()
    const job = cancelJob(Number(params.id))
    return job ? HttpResponse.json(publicJob(job)) : notFound('Задача')
  }),
  http.post('/api/jobs/:id/restart', async ({ params }) => {
    await latency()
    const old = db.jobs.find((j) => j.id === Number(params.id))
    if (!old) return notFound('Задача')
    if (!['failed', 'cancelled'].includes(old.status))
      return HttpResponse.json({ detail: `Задача в статусе ${old.status}` }, { status: 409 })
    const job = restartJob(old.id)
    return job ? HttpResponse.json(publicJob(job)) : HttpResponse.json({ detail: 'Нельзя перезапустить' }, { status: 409 })
  }),

  sse('/api/jobs/:id/events', ({ client, params, request }) => {
    const job = db.jobs.find((j) => j.id === Number(params.id))
    let sent = Number(request.headers.get('last-event-id') ?? 0)
    const pump = () => {
      tick()
      if (!job) return true
      for (const ev of job.events.filter((e) => e.id > sent)) {
        client.send({ id: String(ev.id), event: 'job_event', data: JSON.stringify(ev) } as never)
        sent = ev.id
      }
      if (['done', 'failed', 'cancelled'].includes(job.status)) {
        client.send({ event: 'end', data: JSON.stringify({ status: job.status }) } as never)
        return true
      }
      return false
    }
    if (pump()) return void queueMicrotask(() => client.close())
    const timer = setInterval(() => {
      if (pump()) {
        clearInterval(timer)
        client.close()
      }
    }, 400)
    request.signal.addEventListener('abort', () => clearInterval(timer))
  }),
]
