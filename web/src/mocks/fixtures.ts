/** Static mock fixtures mirroring `mtmct.api.schemas` (used only with VITE_MOCK=1). */
import type { ComponentOut, DatasetOut, ProfileOut, SequenceOut, SystemOut } from '@/api/types.gen'

const previewUrls = (ds: string, seq: string, cams: string[]) =>
  Object.fromEntries(cams.map((c) => [c, `/api/mock-preview/${ds}/${seq}/${c}`]))

const cams = (n: number) => Array.from({ length: n }, (_, i) => `c${i}`)

export function seq(
  ds: string,
  name: string,
  nCams: number,
  duration: number,
  splits: SequenceOut['splits'],
  calibrated: boolean,
): SequenceOut {
  const cameras = cams(nCams)
  return {
    name,
    cameras,
    duration_s: duration,
    splits,
    has_gt: true,
    has_calibration: calibrated,
    preview_urls: previewUrls(ds, name, cameras),
    media_ready: true,
  }
}

/** Sequences a dataset gets once its fetch job completes. */
export const SEQUENCES_AFTER_FETCH: Record<string, SequenceOut[]> = {
  wildtrack: [seq('wildtrack', 'wildtrack', 7, 200, { tune: [0, 0.9], test: [0.9, 1], all: [0, 1] }, true)],
  aic24: [seq('aic24', 'scene_041', 10, 300, { tune: [0, 0.5], test: [0.5, 1] }, true)],
}

export const DATASETS: DatasetOut[] = [
  {
    id: 'epfl_terrace',
    title: 'EPFL Terrace',
    licence: 'research use',
    source_url: 'https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/',
    overlap: 'overlap',
    real: true,
    n_cameras: '4',
    download_gb: 0.2,
    requires_auth: false,
    description: '4 перекрывающиеся камеры 360×288 на террасе, до 9 человек. Smoke-тест и отладка (dev).',
    status: 'ready',
    status_message: '',
    job_id: null,
    sequences: [seq('epfl_terrace', 'terrace1', 4, 210, { all: [0, 1] }, true)],
    custom: false,
  },
  {
    id: 'campus',
    title: 'CAMPUS',
    licence: 'MIT',
    source_url: 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset',
    overlap: 'overlap',
    real: true,
    n_cameras: '3–4',
    download_gb: 0.8,
    requires_auth: false,
    description: 'Сад кампуса, частичное перекрытие, 1080p, калибровка K,R,T. Основной overlap: Garden1 — tune, Garden2 — test.',
    status: 'ready',
    status_message: '',
    job_id: null,
    sequences: [
      seq('campus', 'garden1', 4, 230, { tune: [0, 1] }, true),
      seq('campus', 'garden2', 4, 245, { test: [0, 1] }, true),
    ],
    custom: false,
  },
  {
    id: 'chirla',
    title: 'CHIRLA',
    licence: 'CC BY 4.0',
    source_url: 'https://huggingface.co/datasets/bdager/CHIRLA',
    overlap: 'nonoverlap',
    real: true,
    n_cameras: '7',
    download_gb: 0.72,
    requires_auth: true,
    description: 'Коридоры офиса, камеры без перекрытия, долгие переходы. Основной nonoverlap.',
    status: 'ready',
    status_message: '',
    job_id: null,
    sequences: [
      seq('chirla', 'seq_001', 7, 300, { tune: [0, 1] }, false),
      seq('chirla', 'seq_004', 7, 280, { test: [0, 1] }, false),
    ],
    custom: false,
  },
  {
    id: 'wildtrack',
    title: 'WILDTRACK',
    licence: 'research use (не указана)',
    source_url: 'https://www.epfl.ch/labs/cvlab/data/data-wildtrack/',
    overlap: 'overlap',
    real: true,
    n_cameras: '7',
    download_gb: 6.8,
    requires_auth: false,
    description: 'Площадь ETH, 7 камер с сильным перекрытием; разметка 2 fps. Сравнение с литературой.',
    status: 'absent',
    status_message: '',
    job_id: null,
    sequences: [],
    custom: false,
  },
  {
    id: 'aic24',
    title: 'AI City 2024 · scene_041',
    licence: 'CC BY 4.0',
    source_url: 'https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces',
    overlap: 'overlap',
    real: false,
    n_cameras: '10',
    download_gb: 2.3,
    requires_auth: false,
    description: 'Синтетика, склад, 10 камер. Масштаб и сравнение с AIC.',
    status: 'error',
    status_message: 'Загрузка прервана: таймаут соединения с huggingface.co',
    job_id: null,
    sequences: [],
    custom: false,
  },
]

export const SYSTEM: SystemOut = {
  version: '0.1.0',
  git_sha: 'ac375cd',
  cpu: '12th Gen Intel(R) Core(TM) i5-12450H',
  cpu_threads: 12,
  ram_total_gb: 15.3,
  ram_available_gb: 5.8,
  disk_free_gb: 96.4,
  openvino_devices: ['CPU', 'GPU'],
  licences: [
    { kind: 'model', name: 'YOLOX-s', licence: 'Apache-2.0', note: 'COCO weights' },
    { kind: 'model', name: 'OSNet-AIN x1_0', licence: 'MIT', note: 'MSMT17' },
    { kind: 'model', name: 'OMZ person-reidentification-retail-0287', licence: 'Apache-2.0', note: '' },
    { kind: 'library', name: 'TrackEval', licence: 'MIT', note: 'vendored' },
    { kind: 'library', name: 'trackers (Roboflow)', licence: 'Apache-2.0', note: '' },
    { kind: 'dataset', name: 'CAMPUS', licence: 'MIT', note: '' },
    { kind: 'dataset', name: 'CHIRLA', licence: 'CC BY 4.0', note: 'gated' },
  ],
}

// ---- registry: params schemas exactly as pydantic emits them ----
const num = (title: string, def: number, min?: number, max?: number, description?: string) => ({
  title,
  type: 'number',
  default: def,
  ...(min === undefined ? {} : { minimum: min }),
  ...(max === undefined ? {} : { maximum: max }),
  ...(description ? { description } : {}),
})
const int = (title: string, def: number, min?: number) => ({
  title,
  type: 'integer',
  default: def,
  ...(min === undefined ? {} : { exclusiveMinimum: min - 1 }),
})
const enm = (title: string, def: string | number, values: (string | number)[]) => ({
  title,
  default: def,
  enum: values,
  type: typeof def === 'number' ? 'integer' : 'string',
})
const bool = (title: string, def: boolean) => ({ title, type: 'boolean', default: def })
const params = (properties: Record<string, unknown>) => ({ title: 'Params', type: 'object', properties })

const comp = (
  category: string,
  name: string,
  description: string,
  properties: Record<string, unknown>,
  licence = 'Apache-2.0',
  optional = false,
): ComponentOut => ({ category, name, description, params_schema: params(properties), licence, optional })

const detParams = (size: number) => ({
  conf: num('Conf', 0.3, 0, 1, 'Порог уверенности детекции'),
  nms: num('Nms', 0.45, 0, 1),
  input_size: enm('Input Size', size, [416, 640]),
  device: enm('Device', 'CPU', ['CPU', 'GPU']),
  backend: enm('Backend', 'openvino', ['openvino', 'onnxruntime']),
})
const reidParams = {
  batch: int('Batch', 32, 1),
  device: enm('Device', 'CPU', ['CPU', 'GPU']),
  backend: enm('Backend', 'openvino', ['openvino', 'onnxruntime']),
}
const solverBase = { threshold: num('Threshold', 0.45, 0, 1, 'Порог расстояния слияния кластеров') }

export const REGISTRY: Record<string, ComponentOut[]> = {
  detector: [
    comp('detector', 'yolox_s', 'YOLOX-s, 640 px, OpenVINO', detParams(640)),
    comp('detector', 'yolox_tiny', 'YOLOX-tiny, 416 px — быстрый', detParams(416)),
    comp('detector', 'ultralytics_yolo', 'Ultralytics YOLO11n (опционально)', detParams(640), 'AGPL-3.0', true),
  ],
  tracker: [
    comp('tracker', 'bytetrack', 'ByteTrack (trackers)', {
      track_activation_threshold: num('Track Activation Threshold', 0.25, 0, 1),
      lost_track_buffer: int('Lost Track Buffer', 30, 1),
      minimum_matching_threshold: num('Minimum Matching Threshold', 0.8, 0, 1),
    }),
    comp('tracker', 'botsort', 'BoT-SORT с компенсацией движения камеры', {
      track_high_thresh: num('Track High Thresh', 0.5, 0, 1),
      new_track_thresh: num('New Track Thresh', 0.6, 0, 1),
      cmc: bool('Cmc', false),
    }),
    comp('tracker', 'ocsort', 'OC-SORT', { det_thresh: num('Det Thresh', 0.3, 0, 1), max_age: int('Max Age', 30, 1) }),
  ],
  reid: [
    comp('reid', 'osnet_ain_x1_0', 'OSNet-AIN x1_0 (MSMT17), 512-d', reidParams, 'MIT'),
    comp('reid', 'osnet_x0_25', 'OSNet x0_25 — лёгкая', reidParams, 'MIT'),
    comp('reid', 'omz_0287', 'OpenVINO person-reidentification-retail-0287', reidParams),
  ],
  foot_point: [comp('foot_point', 'bottom_center', 'Низ-центр рамки', {})],
  cost: [
    comp('cost', 'appearance', 'Косинусное расстояние портретов', {
      agg: enm('Agg', 'mean', ['mean', 'max', 'gallery_topk']),
      topk: int('Topk', 5, 1),
    }),
    comp('cost', 'geometry', 'Расстояние траекторий на плоскости пола', {
      gate_m: num('Gate M', 1.5, 0, 10, 'Максимальное расстояние, м'),
      max_speed_mps: num('Max Speed Mps', 3, 0, 10),
    }),
    comp('cost', 'time', 'Окна времени переходов между камерами', {
      hard: bool('Hard', true),
      unseen_penalty: num('Unseen Penalty', 0.5, 0, 1),
    }),
  ],
  solver: [
    comp('solver', 'agglomerative', 'Агломеративная кластеризация с cannot-link', {
      linkage: { $ref: '#/$defs/Linkage', default: 'average' },
      ...solverBase,
    }),
    comp('solver', 'anchor', 'Anchor-guided: длинные треклеты как якоря', {
      ...solverBase,
      anchor_min_len_s: num('Anchor Min Len S', 3, 0, 60),
    }),
    comp('solver', 'hungarian', 'Онлайн: банк признаков + Hungarian', { ...solverBase, online: bool('Online', true) }),
  ],
}
for (const c of REGISTRY.solver) {
  if ('linkage' in (c.params_schema.properties as object))
    c.params_schema.$defs = { Linkage: { enum: ['average', 'complete', 'single'], title: 'Linkage', type: 'string' } }
}

const BASE_CONFIG = {
  profile: 'overlap',
  data: { target_fps: 10, t_start: 0, max_seconds: null, max_rss_gb: 9 },
  detect: { name: 'yolox_s', params: { conf: 0.3, nms: 0.45, input_size: 640 } },
  track: { name: 'bytetrack', params: {} },
  reid: { name: 'osnet_ain_x1_0', params: { batch: 32 } },
  refine: {
    enabled: true,
    split_thr: 0.5,
    window: 10,
    max_gap_s: 1,
    stitch_dist_px: 80,
    stitch_app_thr: 0.6,
    min_len: 8,
    min_score: 0.4,
    interp_gap: 10,
  },
  portrait: { reid_stride: 3, top_k: 10 },
  geometry: { foot_point: { name: 'bottom_center', params: {} }, smooth_window: 5 },
  associate: {
    costs: { appearance: { weight: 1, params: { agg: 'mean' } } } as Record<string, unknown>,
    constraints: { same_camera_overlap: 'cannot_link' },
    rerank: null,
    solver: { name: 'agglomerative', params: { linkage: 'average', threshold: 0.45 } },
  },
  postprocess: { min_global_len: 10, drop_short_single_camera: true },
  eval: { iou: 0.5, thresholds: Array.from({ length: 15 }, (_, i) => Math.round((0.1 + 0.05 * i) * 100) / 100) },
}

export const PROFILES: ProfileOut[] = [
  {
    name: 'overlap',
    description: 'Перекрывающиеся калиброванные камеры: геометрия на плоскости пола + внешний вид.',
    config: {
      ...BASE_CONFIG,
      associate: {
        ...BASE_CONFIG.associate,
        costs: { ...BASE_CONFIG.associate.costs, geometry: { weight: 1, params: { gate_m: 1.5, max_speed_mps: 3 } } },
      },
    },
  },
  {
    name: 'nonoverlap',
    description: 'Камеры без перекрытия: внешний вид + окна времени переходов (topology.json).',
    config: {
      ...BASE_CONFIG,
      profile: 'nonoverlap',
      associate: {
        ...BASE_CONFIG.associate,
        costs: { ...BASE_CONFIG.associate.costs, time: { weight: 0.5, params: { hard: true, unseen_penalty: 0.5 } } },
      },
    },
  },
]

/** Placeholder camera frame (SVG) for previews and crops. */
export function previewSvg(label: string, hue: number, w = 320, h = 180): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="hsl(${hue} 25% 32%)"/><stop offset="1" stop-color="hsl(${(hue + 40) % 360} 30% 18%)"/></linearGradient></defs>
<rect width="100%" height="100%" fill="url(#g)"/>
<path d="M0 ${h * 0.62} L${w} ${h * 0.5} L${w} ${h} L0 ${h} Z" fill="hsl(${hue} 15% 24%)"/>
${[0.22, 0.48, 0.71]
  .map((x, i) => `<rect x="${w * x}" y="${h * (0.38 + i * 0.04)}" width="${w * 0.045}" height="${h * 0.3}" rx="3" fill="hsl(${(hue + i * 70) % 360} 35% 62%)" opacity=".85"/>`)
  .join('')}
<text x="10" y="22" font-family="ui-monospace,monospace" font-size="13" fill="#fff" opacity=".85">${label}</text>
</svg>`
}
