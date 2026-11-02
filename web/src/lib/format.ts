/** Display formatting helpers (Russian locale). */

const dateFmt = new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
const rel = new Intl.RelativeTimeFormat('ru-RU', { numeric: 'auto' })

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : dateFmt.format(d)
}

export function formatRelative(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return '—'
  const s = (new Date(iso).getTime() - now) / 1000
  if (!Number.isFinite(s)) return '—'
  const a = Math.abs(s)
  if (a < 45) return 'только что'
  if (a < 3600) return rel.format(Math.round(s / 60), 'minute')
  if (a < 86400) return rel.format(Math.round(s / 3600), 'hour')
  return rel.format(Math.round(s / 86400), 'day')
}

export function formatEta(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return '—'
  if (s < 60) return `${Math.max(0, Math.round(s))} с`
  const m = Math.floor(s / 60)
  return m < 60 ? `${m} мин ${String(Math.round(s % 60)).padStart(2, '0')} с` : `${Math.floor(m / 60)} ч ${m % 60} мин`
}


export function formatValue(v: unknown): string {
  if (v === null || v === undefined) return 'null'
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(Math.round(v * 1e4) / 1e4)
  if (typeof v === 'string') return v
  return JSON.stringify(v)
}

export const STAGE_LABELS: Record<string, string> = {
  ingest: 'Ingest',
  detect: 'Детекция',
  track: 'Трекинг',
  reid: 'ReID',
  refine: 'Чистка треклетов',
  portrait: 'Портреты',
  geometry: 'Геометрия',
  associate: 'Ассоциация',
  postprocess: 'Постобработка',
  eval: 'Оценка',
  oracle: 'GT-треклеты',
  download: 'Скачивание',
  extract: 'Распаковка',
  proxy: 'Прокси-видео',
}

export const JOB_KIND_LABELS: Record<string, string> = {
  run: 'Прогон',
  sweep: 'Sweep',
  dataset_fetch: 'Скачивание',
  dataset_prepare: 'Подготовка',
  eval_only: 'Переоценка',
}
