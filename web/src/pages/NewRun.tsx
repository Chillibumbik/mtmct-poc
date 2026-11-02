import { useMemo, useState, type ReactNode } from 'react'
import { Link, useNavigate, useSearch } from '@tanstack/react-router'
import { AlertTriangle, Grid3x3, Play, Plus, RotateCcw, Trash2, X } from 'lucide-react'
import { toast } from 'sonner'
import { errorText } from '@/api/client'
import { useCreateRun, useCreateSweep, useDatasets, useProfiles, useRegistry } from '@/api/hooks'
import type { ComponentOut, DatasetOut, ProfileOut } from '@/api/types.gen'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { PageHeader } from '@/components/app/PageHeader'
import { SchemaForm } from '@/components/app/SchemaForm'
import { EmptyState, ErrorState, LoadingBlock } from '@/components/app/States'
import { applyOverrides, deepEqual, getPath, leafKeys, parseGridValues, type Json } from '@/lib/config'
import { formatSeconds } from '@/lib/metrics'
import { formatValue } from '@/lib/format'
import { fieldsFromSchema } from '@/lib/schemaForm'
import { cn } from '@/lib/utils'

const STAGES = [
  { path: 'detect', category: 'detector', title: 'Детектор', fullOnly: true },
  { path: 'track', category: 'tracker', title: 'Трекер в камере', fullOnly: true },
  { path: 'reid', category: 'reid', title: 'ReID-модель', fullOnly: false },
  { path: 'geometry.foot_point', category: 'foot_point', title: 'Точка ног', fullOnly: false },
  { path: 'associate.solver', category: 'solver', title: 'Решатель ассоциации', fullOnly: false },
] as const

const THRESHOLD_KEY = 'associate.solver.params.threshold'
const RERANK_DEFAULT = { k1: 20, k2: 6, lambda_: 0.3 }

/** Rough wall-time guess: ~5.5 frames/s for yolox_s + OSNet on the reference laptop (spec §11). */
function estimateSeconds(cfg: Json, cams: number, durationS: number, mode: string): number {
  const fps = Number(getPath(cfg, 'data.target_fps') ?? 10)
  const cap = getPath(cfg, 'data.max_seconds')
  const dur = typeof cap === 'number' ? Math.min(cap, durationS) : durationS
  const frames = dur * fps * cams
  if (mode === 'oracle') return 20 + frames * 0.01
  return 15 + frames / (getPath(cfg, 'detect.name') === 'yolox_tiny' ? 9 : 5.5)
}

interface Editor {
  cfg: Json
  base: Json
  overrides: Json
  set: (key: string, value: unknown) => void
  setComponent: (path: string, comp: ComponentOut) => void
  setParam: (path: string, key: string, value: unknown) => void
}

function useEditor(base: Json): Editor & { reset: (key?: string) => void } {
  const [overrides, setOverrides] = useState<Json>({})
  const cfg = useMemo(() => applyOverrides(base, overrides), [base, overrides])
  const update = (fn: (o: Json) => void) =>
    setOverrides((prev) => {
      const next = { ...prev }
      fn(next)
      return next
    })
  return {
    cfg,
    base,
    overrides,
    set: (key, value) =>
      update((o) => {
        if (deepEqual(getPath(base, key), value)) delete o[key]
        else o[key] = value
      }),
    setComponent: (path, comp) =>
      update((o) => {
        for (const k of Object.keys(o)) if (k.startsWith(`${path}.`)) delete o[k]
        if (getPath(base, `${path}.name`) === comp.name) return
        o[`${path}.name`] = comp.name
        // Params the new component does not know are dropped; missing ones take its own defaults (pydantic).
        const cur = (getPath(base, `${path}.params`) ?? {}) as Json
        const known = fieldsFromSchema(comp.params_schema).map((f) => f.key)
        const kept = Object.fromEntries(Object.entries(cur).filter(([k]) => known.includes(k)))
        if (!deepEqual(kept, cur)) o[`${path}.params`] = kept
      }),
    setParam: (path, key, value) =>
      update((o) => {
        const whole = o[`${path}.params`]
        if (whole && typeof whole === 'object') o[`${path}.params`] = { ...(whole as Json), [key]: value }
        else if (deepEqual(getPath(base, `${path}.params.${key}`), value)) delete o[`${path}.params.${key}`]
        else o[`${path}.params.${key}`] = value
      }),
    reset: (key) => setOverrides((prev) => (key ? Object.fromEntries(Object.entries(prev).filter(([k]) => k !== key)) : {})),
  }
}

function Row({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="grid items-start gap-1.5 sm:grid-cols-[11rem_1fr] sm:gap-4">
      <div className="pt-1.5 text-sm">
        {label}
        {hint && <div className="text-xs text-muted-foreground">{hint}</div>}
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  )
}

function StageEditor({ ed, stage, components, disabled }: { ed: Editor; stage: (typeof STAGES)[number]; components: ComponentOut[]; disabled: boolean }) {
  const name = String(getPath(ed.cfg, `${stage.path}.name`) ?? '')
  const comp = components.find((c) => c.name === name)
  const params = (getPath(ed.cfg, `${stage.path}.params`) ?? {}) as Json
  const baseParams = getPath(ed.base, `${stage.path}.name`) === name ? ((getPath(ed.base, `${stage.path}.params`) ?? {}) as Json) : undefined
  return (
    <div className={cn('space-y-3 rounded-lg border p-3', disabled && 'opacity-50')}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="w-44 text-sm font-medium">{stage.title}</span>
        <Select value={name} onValueChange={(v) => ed.setComponent(stage.path, components.find((c) => c.name === v)!)} disabled={disabled}>
          <SelectTrigger size="sm" className="w-56 font-mono text-xs">
            <SelectValue placeholder="—" />
          </SelectTrigger>
          <SelectContent>
            {components.map((c) => (
              <SelectItem key={c.name} value={c.name} className="font-mono text-xs">
                {c.name}
                {c.optional && <span className="text-muted-foreground"> · опц.</span>}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {comp && (
          <>
            <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground" title={comp.description}>
              {comp.description}
            </span>
            <Badge variant="outline" className={cn('font-mono font-normal', comp.licence.includes('GPL') && 'text-warn')}>
              {comp.licence}
            </Badge>
          </>
        )}
        {disabled && <span className="text-xs text-muted-foreground">не используется в oracle-режиме</span>}
      </div>
      {comp && !disabled && (
        <SchemaForm
          schema={comp.params_schema}
          values={params}
          defaults={baseParams}
          exclude={stage.path === 'associate.solver' ? ['threshold'] : []}
          onChange={(k, v) => ed.setParam(stage.path, k, v)}
        />
      )}
      {!comp && name && <div className="text-xs text-warn">Компонент «{name}» не найден в реестре</div>}
    </div>
  )
}

function CostsEditor({ ed, costs }: { ed: Editor; costs: ComponentOut[] }) {
  const current = (getPath(ed.cfg, 'associate.costs') ?? {}) as Record<string, { weight?: number; params?: Json }>
  return (
    <div className="space-y-2">
      {costs.map((c) => {
        const spec = current[c.name]
        const weight = spec?.weight ?? 0
        const on = !!spec && weight > 0
        const key = `associate.costs.${c.name}`
        const baseSpec = getPath(ed.base, key) as { params?: Json } | undefined
        return (
          <div key={c.name} className="space-y-3 rounded-lg border p-3">
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex w-44 items-center gap-2 text-sm font-medium">
                <Switch checked={on} onCheckedChange={(v) => ed.set(`${key}.weight`, v ? (getPath(ed.base, `${key}.weight`) as number) || 1 : 0)} />
                <span className="font-mono">{c.name}</span>
              </label>
              <div className="flex min-w-48 flex-1 items-center gap-3">
                <span className="text-xs text-muted-foreground">вес</span>
                <Slider
                  className="max-w-64"
                  min={0}
                  max={2}
                  step={0.05}
                  value={[weight]}
                  disabled={!on}
                  onValueChange={([v]) => ed.set(`${key}.weight`, Math.round(v * 100) / 100)}
                  aria-label={`Вес стоимости ${c.name}`}
                />
                <span className="w-10 text-right font-mono text-xs tabular-nums">{weight.toFixed(2)}</span>
              </div>
              <span className="text-xs text-muted-foreground">{c.description}</span>
            </div>
            {on && <SchemaForm schema={c.params_schema} values={spec?.params ?? {}} defaults={baseSpec?.params} onChange={(k, v) => ed.set(`${key}.params.${k}`, v)} />}
          </div>
        )
      })}
    </div>
  )
}

function Summary({
  ed,
  reset,
  estimate,
  children,
}: {
  ed: Editor
  reset: (key?: string) => void
  estimate: number | null
  children: ReactNode
}) {
  const entries = Object.entries(ed.overrides)
  return (
    <Card className="sticky top-5 gap-3 py-4">
      <CardHeader className="flex flex-row items-center justify-between px-4">
        <CardTitle className="text-sm">Изменения относительно профиля</CardTitle>
        {entries.length > 0 && (
          <Button variant="ghost" size="xs" onClick={() => reset()}>
            <RotateCcw /> Сбросить
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-3 px-4">
        {entries.length === 0 ? (
          <div className="text-xs text-muted-foreground">Базовый конфиг профиля без изменений.</div>
        ) : (
          <ul className="space-y-1">
            {entries.map(([k, v]) => (
              <li key={k} className="group flex items-start gap-2 font-mono text-xs">
                <span className="min-w-0 flex-1 break-all">
                  <span className="text-muted-foreground">{k}</span> = <span className="text-brand">{formatValue(v)}</span>
                </span>
                <button className="opacity-40 hover:opacity-100" aria-label={`Сбросить ${k}`} onClick={() => reset(k)}>
                  <X className="size-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
        {estimate !== null && (
          <div className="border-t pt-3 text-xs text-muted-foreground">
            Оценка времени: <b className="text-foreground tabular-nums">≈ {formatSeconds(estimate)}</b> (ориентир; при готовом кеше быстрее)
          </div>
        )}
        {children}
      </CardContent>
    </Card>
  )
}

interface GridRow {
  key: string
  values: string
}

function SweepEditor({ rows, setRows, keys }: { rows: GridRow[]; setRows: (r: GridRow[]) => void; keys: string[] }) {
  return (
    <div className="space-y-2">
      <datalist id="config-keys">
        {keys.map((k) => (
          <option key={k} value={k} />
        ))}
      </datalist>
      {rows.map((r, i) => {
        const n = parseGridValues(r.values).length
        return (
          <div key={i} className="space-y-1 rounded-md border p-2">
            <div className="flex items-center gap-1">
              <Input
                list="config-keys"
                className="h-7 flex-1 font-mono text-xs"
                placeholder="associate.solver.params.threshold"
                value={r.key}
                onChange={(e) => setRows(rows.map((x, j) => (j === i ? { ...x, key: e.target.value } : x)))}
                aria-label="Ключ конфига"
              />
              <Button variant="ghost" size="icon-xs" aria-label="Удалить строку" onClick={() => setRows(rows.filter((_, j) => j !== i))}>
                <Trash2 />
              </Button>
            </div>
            <div className="flex items-center gap-2">
              <Input
                className="h-7 flex-1 font-mono text-xs"
                placeholder="0.35, 0.45, 0.55"
                value={r.values}
                onChange={(e) => setRows(rows.map((x, j) => (j === i ? { ...x, values: e.target.value } : x)))}
                aria-label="Значения через запятую"
              />
              <span className="w-8 text-right text-xs tabular-nums text-muted-foreground">×{n}</span>
            </div>
          </div>
        )
      })}
      <Button variant="outline" size="sm" onClick={() => setRows([...rows, { key: '', values: '' }])}>
        <Plus /> Параметр
      </Button>
    </div>
  )
}

function Form({ datasets, profiles, registry }: { datasets: DatasetOut[]; profiles: ProfileOut[]; registry: Record<string, ComponentOut[]> }) {
  const search = useSearch({ from: '/runs/new' })
  const navigate = useNavigate({ from: '/runs/new' })
  const ready = datasets.filter((d) => d.status === 'ready' && d.sequences.length > 0)
  const ds = ready.find((d) => d.id === search.dataset) ?? ready[0]
  const seq = ds?.sequences.find((s) => s.name === search.sequence) ?? ds?.sequences[0]
  const tab = search.tab ?? 'single'
  const splits = seq ? Object.keys(seq.splits) : []
  const split = search.split && (splits.includes(search.split) || search.split === 'all') ? search.split : (splits[0] ?? 'all')
  const mode = search.mode ?? 'full'
  const defaultProfile = ds?.overlap === 'nonoverlap' ? 'nonoverlap' : 'overlap'
  const [profileChoice, setProfileChoice] = useState<string>('auto')
  const profileName = profileChoice === 'auto' ? defaultProfile : profileChoice
  const profile = profiles.find((p) => p.name === profileName) ?? profiles[0]
  const [name, setName] = useState('')
  const [rows, setRows] = useState<GridRow[]>([{ key: THRESHOLD_KEY, values: '0.35, 0.45, 0.55' }])
  const ed = useEditor(profile.config as Json)
  const createRun = useCreateRun()
  const createSweep = useCreateSweep()

  const setSearch = (patch: Partial<typeof search>) => void navigate({ search: (s) => ({ ...s, ...patch }), replace: true })
  const frac = seq && split !== 'all' && seq.splits[split] ? seq.splits[split][1] - seq.splits[split][0] : 1
  const estimate = seq ? estimateSeconds(ed.cfg, seq.cameras.length, seq.duration_s * frac, mode) : null
  const grid = Object.fromEntries(rows.filter((r) => r.key.trim()).map((r) => [r.key.trim(), parseGridValues(r.values)]))
  const nRuns = Object.keys(grid).length ? Object.values(grid).reduce((a, v) => a * v.length, 1) : 0
  const configKeys = useMemo(() => leafKeys(ed.cfg), [ed.cfg])
  const threshold = Number(getPath(ed.cfg, THRESHOLD_KEY) ?? 0.45)
  const rerank = getPath(ed.cfg, 'associate.rerank')

  if (!ds || !seq) {
    return (
      <EmptyState
        title="Нет подготовленных датасетов"
        hint="Скачайте и подготовьте датасет или загрузите свои видео."
        action={
          <Button asChild size="sm" variant="outline">
            <Link to="/data">К данным</Link>
          </Button>
        }
      />
    )
  }

  const common = {
    dataset: ds.id,
    sequence: seq.name,
    split,
    mode,
    profile: profileChoice === 'auto' ? null : (profileChoice as 'overlap' | 'nonoverlap'),
    name: name.trim() || null,
  }
  const submitRun = () =>
    createRun.mutate(
      { ...common, overrides: ed.overrides },
      {
        onSuccess: (run) => {
          toast.success(`Прогон #${run.id} поставлен в очередь`)
          void navigate({ to: '/jobs', search: { job: run.job_id ?? undefined } })
        },
        onError: (e) => toast.error(errorText(e)),
      },
    )
  const submitSweep = () =>
    createSweep.mutate(
      { ...common, base_overrides: ed.overrides, grid },
      {
        onSuccess: (sw) => {
          toast.success(`Sweep #${sw.id}: ${sw.n_runs} прогонов в очереди`)
          void navigate({ to: '/experiments', search: { sweep: sw.id } })
        },
        onError: (e) => toast.error(errorText(e)),
      },
    )
  const badGrid = rows.some((r) => r.key.trim() && parseGridValues(r.values).length === 0)

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[1fr_22rem]">
      <div className="min-w-0 space-y-5">
        <Card className="gap-4 py-4">
          <CardHeader className="px-4">
            <CardTitle className="text-sm">Данные и режим</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 px-4">
            <Row label="Датасет">
              <Select value={ds.id} onValueChange={(v) => setSearch({ dataset: v, sequence: undefined, split: undefined })}>
                <SelectTrigger size="sm" className="w-72">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ready.map((d) => (
                    <SelectItem key={d.id} value={d.id}>
                      {d.title} <span className="text-muted-foreground">· {d.overlap}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Row>
            <Row label="Последовательность" hint={`${seq.cameras.length} камер · ${formatSeconds(seq.duration_s)}`}>
              <Select value={seq.name} onValueChange={(v) => setSearch({ sequence: v, split: undefined })}>
                <SelectTrigger size="sm" className="w-72 font-mono">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ds.sequences.map((s) => (
                    <SelectItem key={s.name} value={s.name} className="font-mono">
                      {s.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Row>
            <Row label="Сплит" hint="Подбор параметров — только tune">
              <ToggleGroup type="single" variant="outline" size="sm" value={split} onValueChange={(v) => v && setSearch({ split: v })}>
                {[...new Set([...splits, 'all'])].map((s) => (
                  <ToggleGroupItem key={s} value={s} className="px-3 font-mono">
                    {s}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </Row>
            <Row label="Режим" hint={mode === 'oracle' ? 'Треклеты из GT: проверяется только ассоциация' : 'Полный пайплайн от видео'}>
              <ToggleGroup type="single" variant="outline" size="sm" value={mode} onValueChange={(v) => v && setSearch({ mode: v as 'full' | 'oracle' })}>
                <ToggleGroupItem value="full" className="px-3">
                  full
                </ToggleGroupItem>
                <ToggleGroupItem value="oracle" className="px-3" disabled={!seq.has_gt}>
                  oracle
                </ToggleGroupItem>
              </ToggleGroup>
            </Row>
            <Row label="Профиль" hint={profile.description}>
              <Select value={profileChoice} onValueChange={setProfileChoice}>
                <SelectTrigger size="sm" className="w-72">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">по датасету ({defaultProfile})</SelectItem>
                  {profiles.map((p) => (
                    <SelectItem key={p.name} value={p.name}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {profileName === 'overlap' && !seq.has_calibration && (
                <div className="mt-1.5 flex items-center gap-1.5 text-xs text-warn">
                  <AlertTriangle className="size-3.5" /> У последовательности нет калибровки — стоимость geometry будет отключена.
                </div>
              )}
            </Row>
            <Row label="Название" hint="необязательно">
              <Input className="h-8 w-72" value={name} placeholder={`${ds.id}/${seq.name} · ${mode}`} onChange={(e) => setName(e.target.value)} />
            </Row>
          </CardContent>
        </Card>

        <Card className="gap-4 py-4">
          <CardHeader className="px-4">
            <CardTitle className="text-sm">Стадии и компоненты</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 px-4">
            <Row label="Обработка" hint="data.*">
              <div className="flex flex-wrap items-center gap-4">
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  target_fps
                  <Input
                    type="number"
                    min={1}
                    max={30}
                    className="h-8 w-20 font-mono text-xs"
                    value={String(getPath(ed.cfg, 'data.target_fps') ?? 10)}
                    onChange={(e) => e.target.value && ed.set('data.target_fps', Number(e.target.value))}
                  />
                </label>
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  max_seconds
                  <Input
                    type="number"
                    min={1}
                    className="h-8 w-24 font-mono text-xs"
                    placeholder="весь"
                    value={getPath(ed.cfg, 'data.max_seconds') == null ? '' : String(getPath(ed.cfg, 'data.max_seconds'))}
                    onChange={(e) => ed.set('data.max_seconds', e.target.value ? Number(e.target.value) : null)}
                  />
                </label>
                <label className="flex items-center gap-2 text-xs">
                  <Switch checked={Boolean(getPath(ed.cfg, 'refine.enabled'))} onCheckedChange={(v) => ed.set('refine.enabled', v)} />
                  refine (чистка треклетов)
                </label>
              </div>
            </Row>
            {STAGES.map((s) => (
              <StageEditor key={s.path} ed={ed} stage={s} components={registry[s.category] ?? []} disabled={s.fullOnly && mode === 'oracle'} />
            ))}
          </CardContent>
        </Card>

        <Card className="gap-4 py-4">
          <CardHeader className="px-4">
            <CardTitle className="text-sm">Ассоциация: стоимости и порог</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 px-4">
            <Row label="Порог решателя" hint="ниже — меньше склеек, больше дроблений">
              <div className="flex max-w-md items-center gap-3">
                <Slider min={0.05} max={0.95} step={0.01} value={[threshold]} onValueChange={([v]) => ed.set(THRESHOLD_KEY, Math.round(v * 100) / 100)} aria-label="Порог решателя" />
                <span className="w-10 text-right font-mono text-xs tabular-nums">{threshold.toFixed(2)}</span>
              </div>
            </Row>
            <Row label="k-reciprocal re-ranking" hint="associate.rerank">
              <Switch checked={rerank != null} onCheckedChange={(v) => ed.set('associate.rerank', v ? RERANK_DEFAULT : null)} />
            </Row>
            <CostsEditor ed={ed} costs={registry.cost ?? []} />
          </CardContent>
        </Card>
      </div>

      <Summary ed={ed} reset={ed.reset} estimate={estimate === null ? null : tab === 'sweep' ? estimate * Math.max(1, nRuns) : estimate}>
        <Tabs value={tab} onValueChange={(v) => setSearch({ tab: v as 'single' | 'sweep' })}>
          <TabsList className="w-full">
            <TabsTrigger value="single">
              <Play /> Один прогон
            </TabsTrigger>
            <TabsTrigger value="sweep">
              <Grid3x3 /> Sweep
            </TabsTrigger>
          </TabsList>
          <TabsContent value="single" className="pt-2">
            <Button className="w-full" onClick={submitRun} disabled={createRun.isPending}>
              <Play /> Запустить
            </Button>
          </TabsContent>
          <TabsContent value="sweep" className="space-y-3 pt-2">
            <p className="text-xs text-muted-foreground">Декартово произведение значений; изменения выше — общая база для всех прогонов.</p>
            <SweepEditor rows={rows} setRows={setRows} keys={configKeys} />
            {split === 'test' && (
              <div className="flex gap-1.5 text-xs text-warn">
                <AlertTriangle className="size-3.5 shrink-0" /> Перебор на test-сплите нарушает протокол: подбирайте параметры на tune.
              </div>
            )}
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Прогонов</span>
              <b className="tabular-nums">{nRuns}</b>
            </div>
            <Button className="w-full" onClick={submitSweep} disabled={createSweep.isPending || nRuns === 0 || badGrid}>
              <Grid3x3 /> Запустить sweep ({nRuns})
            </Button>
          </TabsContent>
        </Tabs>
      </Summary>
    </div>
  )
}

export function NewRunPage() {
  const datasets = useDatasets()
  const profiles = useProfiles()
  const registry = useRegistry()
  const failed = [datasets, profiles, registry].find((q) => q.isError)
  return (
    <>
      <PageHeader title="Новый прогон" description="Выбор данных, компонентов по стадиям и параметров; sweep — сетка гипотез" />
      {failed ? (
        <ErrorState error={failed.error} onRetry={() => void failed.refetch()} />
      ) : datasets.data && profiles.data && registry.data ? (
        profiles.data.length === 0 ? (
          <EmptyState title="Сервер не вернул профилей конфигурации" />
        ) : (
          <Form datasets={datasets.data} profiles={profiles.data} registry={registry.data.components} />
        )
      ) : (
        <LoadingBlock rows={8} />
      )}
    </>
  )
}
