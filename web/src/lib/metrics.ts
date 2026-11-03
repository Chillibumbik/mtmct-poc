import type { KeyMetrics } from '@/api/types.gen'

/** How a value is rendered: `pct` = ratio 0..1 shown as percent points (62.3). */
export type MetricFormat = 'pct' | 'int' | 'float2' | 'float3' | 'float1' | 'gb' | 'seconds' | 'ms' | 'mb'

export interface MetricDef {
  key: string
  label: string
  description: string
  /** `null` — neither direction is "better" (e.g. counts compared to GT). */
  higherIsBetter: boolean | null
  format: MetricFormat
}

const d = (
  key: string,
  label: string,
  description: string,
  higherIsBetter: boolean | null,
  format: MetricFormat = 'pct',
): MetricDef => ({ key, label, description, higherIsBetter, format })

const DEFS: MetricDef[] = [
  // TrackEval: HOTA family (§9.1, §9.2)
  d('HOTA', 'HOTA', 'Higher Order Tracking Accuracy: среднее геометрическое качества детекции и ассоциации по порогам IoU. Главная сбалансированная метрика трекинга.', true),
  d('DetA', 'DetA', 'Detection Accuracy: насколько точно найдены люди (детекция) без учёта идентичностей.', true),
  d('AssA', 'AssA', 'Association Accuracy: насколько последовательно сохраняются идентичности вдоль треков. Для межкамерной оценки — качество склейки между камерами.', true),
  d('LocA', 'LocA', 'Localization Accuracy: средняя точность рамок (IoU) у совпавших детекций.', true),
  // Identity family
  d('IDF1', 'IDF1', 'Доля правильно идентифицированных рамок (F1 между IDP и IDR) при оптимальном сопоставлении ID. Ключевая метрика сохранения идентичности.', true),
  d('IDP', 'IDP', 'ID Precision: доля предсказанных рамок с правильным ID.', true),
  d('IDR', 'IDR', 'ID Recall: доля GT-рамок, найденных с правильным ID.', true),
  // CLEAR MOT
  d('MOTA', 'MOTA', 'Multi-Object Tracking Accuracy: 1 − (FP + FN + IDSW) / GT. Сильнее всего зависит от детекции; может быть отрицательной.', true),
  d('MOTP', 'MOTP', 'Multi-Object Tracking Precision: средняя точность локализации совпавших рамок.', true),
  d('IDSW', 'IDSW', 'ID switches: сколько раз у GT-человека сменился предсказанный ID.', false, 'int'),
  d('FP', 'FP', 'False positives: лишние рамки, не совпавшие ни с одним GT.', false, 'int'),
  d('FN', 'FN', 'False negatives: пропущенные GT-рамки.', false, 'int'),
  d('Frag', 'Frag', 'Фрагментации: сколько раз GT-трек прерывался и возобновлялся.', false, 'int'),
  // Identity / clustering (§9.3): items are predicted tracklets, each labelled by the majority GT person of its matched boxes
  d('pairwise_p', 'Pairwise P', 'Попарная точность: доля пар треклетов с одним глобальным ID, которые действительно один человек.', true),
  d('pairwise_r', 'Pairwise R', 'Попарная полнота: доля пар треклетов одного человека, получивших один глобальный ID.', true),
  d('pairwise_f1', 'Pairwise F1', 'F1 попарной точности и полноты кластеризации треклетов по глобальным ID. Каждый треклет считается один раз (независимо от длины), человек треклета — большинство его GT-рамок.', true),
  d('bcubed_p', 'BCubed P', 'BCubed precision: средняя по треклетам чистота их кластера.', true),
  d('bcubed_r', 'BCubed R', 'BCubed recall: средняя по треклетам доля «своих» треклетов в их кластере.', true),
  d('bcubed_f', 'BCubed F', 'F-мера BCubed precision/recall по треклетам — устойчивая к размеру кластеров оценка кластеризации.', true),
  d('purity', 'Purity', 'Чистота: доля треклетов, принадлежащих преобладающему GT-человеку своего кластера.', true),
  d('inverse_purity', 'Inverse purity', 'Обратная чистота: доля треклетов человека, попавших в его основной кластер.', true),
  d('n_pred_ids', 'Pred IDs', 'Число предсказанных глобальных ID. Сравнивайте с числом GT-людей: больше — дробления, меньше — склейки.', null, 'int'),
  d('n_gt_ids', 'GT IDs', 'Число людей в разметке (GT).', null, 'int'),
  d('false_merges', 'False merges', 'Ложные склейки: глобальные ID, содержащие заметную долю рамок (≥ max(5, 10 %)) больше чем одного GT-человека.', false, 'int'),
  d('false_splits', 'False splits', 'Ложные дробления: GT-люди, заметная доля рамок которых (≥ max(5, 10 %)) разнесена по нескольким глобальным ID.', false, 'int'),
  d('handover_success', 'Handover', 'Доля переходов человека между камерами, при которых глобальный ID сохранился.', true),
  // ReID sanity (§9.4)
  d('rank1', 'Rank-1', 'CMC rank-1: доля запросов (портрет треклета), у которых ближайший треклет из других камер — тот же человек.', true),
  d('rank5', 'Rank-5', 'CMC rank-5: правильный человек среди 5 ближайших треклетов других камер.', true),
  d('mAP', 'mAP', 'Mean Average Precision поиска того же человека среди треклетов других камер.', true),
  // Threshold curve
  d('threshold', 'Порог', 'Порог расстояния решателя ассоциации: ниже — меньше склеек, больше дроблений.', null, 'float2'),
  // Resources (§9.6)
  d('fps', 'FPS', 'Обработанные кадры в секунду (суммарно по камерам).', true, 'float1'),
  d('peak_rss_gb', 'Peak RAM', 'Пиковая резидентная память процесса воркера (RSS). Цель ≤ 6 ГБ, жёсткий предел 10 ГБ.', false, 'gb'),
  d('wall_time_s', 'Время', 'Полное время прогона.', false, 'seconds'),
  d('cpu_mean', 'CPU, %', 'Средняя загрузка CPU за прогон, %.', null, 'float1'),
  d('p50_ms', 'p50', 'Медианная латентность (детектор — на кадр, ReID — на рамку).', false, 'ms'),
  d('p95_ms', 'p95', '95-й перцентиль латентности.', false, 'ms'),
  d('cache_mb', 'Кеш', 'Размер кеша тяжёлых стадий.', null, 'mb'),
]

const BY_KEY = new Map(DEFS.map((m) => [m.key, m]))

/** Aliases: KeyMetrics field → base metric key (with scope-specific label). */
const KEY_METRIC_DEFS: Record<keyof KeyMetrics, MetricDef> = {
  cross_hota: { ...BY_KEY.get('HOTA')!, key: 'cross_hota', label: 'HOTA×' },
  cross_idf1: { ...BY_KEY.get('IDF1')!, key: 'cross_idf1', label: 'IDF1×' },
  cross_assa: { ...BY_KEY.get('AssA')!, key: 'cross_assa', label: 'AssA×' },
  per_camera_hota: { ...BY_KEY.get('HOTA')!, key: 'per_camera_hota', label: 'HOTA кам.' },
  per_camera_idf1: { ...BY_KEY.get('IDF1')!, key: 'per_camera_idf1', label: 'IDF1 кам.' },
  pairwise_f1: BY_KEY.get('pairwise_f1')!,
  bcubed_f: BY_KEY.get('bcubed_f')!,
  false_merges: BY_KEY.get('false_merges')!,
  false_splits: BY_KEY.get('false_splits')!,
  handover_success: BY_KEY.get('handover_success')!,
  reid_rank1: { ...BY_KEY.get('rank1')!, key: 'reid_rank1' },
  fps: BY_KEY.get('fps')!,
  peak_rss_gb: BY_KEY.get('peak_rss_gb')!,
  wall_time_s: BY_KEY.get('wall_time_s')!,
}
for (const def of Object.values(KEY_METRIC_DEFS)) if (!BY_KEY.has(def.key)) BY_KEY.set(def.key, def)

export const KEY_METRIC_KEYS = Object.keys(KEY_METRIC_DEFS) as (keyof KeyMetrics)[]

/** Metric keys by section of metrics.json, in display order. */
export const METRIC_SECTIONS = {
  per_camera: ['HOTA', 'DetA', 'AssA', 'LocA', 'IDF1', 'IDP', 'IDR', 'MOTA', 'MOTP', 'IDSW', 'FP', 'FN', 'Frag'],
  cross_camera: ['HOTA', 'AssA', 'IDF1', 'IDP', 'IDR', 'MOTA'],
  identity: [
    'pairwise_p',
    'pairwise_r',
    'pairwise_f1',
    'bcubed_p',
    'bcubed_r',
    'bcubed_f',
    'purity',
    'inverse_purity',
    'n_pred_ids',
    'n_gt_ids',
    'false_merges',
    'false_splits',
    'handover_success',
  ],
  reid: ['rank1', 'rank5', 'mAP'],
  threshold_curve: ['threshold', 'pairwise_p', 'pairwise_r', 'IDF1'],
} as const

export function metricDef(key: string): MetricDef | undefined {
  return BY_KEY.get(key)
}

export function directionHint(def: MetricDef): string {
  if (def.higherIsBetter === null) return ''
  return def.higherIsBetter ? '↑ лучше' : '↓ лучше'
}

const MINUS = '−'
const group = (n: number) => Math.round(n).toLocaleString('ru-RU').replace(/\s/g, ' ')
const signed = (s: string) => s.replace('-', MINUS)

export function formatSeconds(s: number): string {
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} с`
  const m = Math.floor(s / 60)
  const rest = Math.round(s % 60)
  if (m < 60) return `${m} мин ${String(rest).padStart(2, '0')} с`
  return `${Math.floor(m / 60)} ч ${String(m % 60).padStart(2, '0')} мин`
}

function render(format: MetricFormat, v: number): string {
  switch (format) {
    case 'pct':
      return signed((v * 100).toFixed(1))
    case 'int':
      return signed(group(v))
    case 'float2':
      return signed(v.toFixed(2))
    case 'float3':
      return signed(v.toFixed(3))
    case 'float1':
      return v.toFixed(1)
    case 'gb':
      return `${v.toFixed(2)} ГБ`
    case 'mb':
      return v >= 1024 ? `${(v / 1024).toFixed(2)} ГБ` : `${Math.round(v)} МБ`
    case 'ms':
      return `${v.toFixed(v < 10 ? 1 : 0)} мс`
    case 'seconds':
      return formatSeconds(v)
  }
}

/** Formats any metric value; unknown keys fall back to integer / 3-decimal rendering. */
export function formatMetric(key: string, value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const def = BY_KEY.get(key)
  if (def) return render(def.format, value)
  return Number.isInteger(value) ? signed(group(value)) : signed(value.toFixed(3))
}

/** Index of the best value (by metric direction) or -1 when undefined. */
export function bestIndex(key: string, values: (number | null | undefined)[]): number {
  const dir = BY_KEY.get(key)?.higherIsBetter
  if (dir === null || dir === undefined) return -1
  let best = -1
  values.forEach((v, i) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return
    const b = best < 0 ? undefined : values[best]
    if (b === undefined || b === null || (dir ? v > b : v < b)) best = i
  })
  return best
}
