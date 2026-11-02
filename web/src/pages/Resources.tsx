import { useCallback, useMemo } from 'react'
import { Link, useNavigate, useSearch } from '@tanstack/react-router'
import type { EChartsOption } from 'echarts'
import { useDatasets, useManyRunMetrics, useProfiles, useRuns } from '@/api/hooks'
import type { ProfileOut, RunOut } from '@/api/types.gen'
import { Card, CardContent } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Chart } from '@/components/app/Chart'
import type { ChartTheme } from '@/lib/chart'
import { MetricLabel } from '@/components/app/Metric'
import { PageHeader, Section } from '@/components/app/PageHeader'
import { EmptyState, LoadingBlock, QueryView } from '@/components/app/States'
import { applyOverrides, getPath, type Json } from '@/lib/config'
import { formatMetric, formatSeconds } from '@/lib/metrics'
import { parseRuntime, type RuntimeSummary } from '@/lib/runtime'

const MAX_RUNS = 20
const ALL = '__all__'

/** ≤ 8 categorical slots: minor stages are folded into groups (fixed order = fixed colours). */
const STAGE_GROUPS: [string, string[]][] = [
  ['Детекция', ['detect']],
  ['Трекинг', ['track', 'oracle']],
  ['ReID', ['reid']],
  ['Чистка', ['refine']],
  ['Ассоциация', ['portrait', 'geometry', 'associate', 'postprocess']],
  ['Оценка', ['eval']],
  ['Прочее', []],
]
const groupOf = (stage: string) => STAGE_GROUPS.find(([, s]) => s.includes(stage))?.[0] ?? 'Прочее'

interface Row {
  run: RunOut
  rt: RuntimeSummary
  label: string
  detector: string
  reid: string
}

function componentName(run: RunOut, profiles: ProfileOut[], path: string): string {
  const base = (profiles.find((p) => p.name === run.profile)?.config ?? {}) as Json
  return String(getPath(applyOverrides(base, run.overrides), path) ?? '?')
}

function StageStack({ rows }: { rows: Row[] }) {
  const option = useCallback(
    (t: ChartTheme): EChartsOption => ({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => formatSeconds(Number(v) * 60) },
      grid: { left: 8, right: 16, top: 36, bottom: 4, containLabel: true },
      xAxis: { type: 'value', axisLabel: { formatter: (v: number) => (v ? `${v} мин` : '0') } },
      yAxis: { type: 'category', inverse: true, data: rows.map((r) => r.label), splitLine: { show: false } },
      series: STAGE_GROUPS.map(([group], i) => ({
        type: 'bar' as const,
        name: group,
        stack: 'time',
        barMaxWidth: 16,
        itemStyle: { color: t.palette[i], borderColor: t.surface, borderWidth: 1 },
        emphasis: { focus: 'series' as const },
        data: rows.map((r) => r.rt.stages.filter((s) => groupOf(s.stage) === group).reduce((a, s) => a + s.wallS, 0) / 60 || null),
      })).filter((s) => s.data.some((v) => v)),
    }),
    [rows],
  )
  return <Chart option={option} height={60 + rows.length * 26} ariaLabel="Время стадий по прогонам в минутах, накопленные столбцы" />
}

function RunBars({ rows, metric, unit, marks }: { rows: Row[]; metric: 'fps' | 'peak_rss_gb'; unit: string; marks?: [number, string][] }) {
  const option = useCallback(
    (t: ChartTheme): EChartsOption => ({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => `${Number(v).toFixed(2)} ${unit}` },
      grid: { left: 8, right: 40, top: marks ? 24 : 8, bottom: 4, containLabel: true },
      xAxis: { type: 'value', max: marks ? (v: { max: number }) => Math.max(v.max, ...marks.map(([m]) => m)) * 1.05 : undefined },
      yAxis: { type: 'category', inverse: true, data: rows.map((r) => r.label), splitLine: { show: false } },
      series: [
        {
          type: 'bar',
          barMaxWidth: 14,
          itemStyle: { color: t.palette[0], borderRadius: [0, 4, 4, 0] },
          label: { show: true, position: 'right', color: t.muted, fontSize: 11, formatter: (p) => Number(p.value).toFixed(1) },
          data: rows.map((r) => (metric === 'fps' ? r.rt.fps : r.rt.peakRssGb)),
          markLine: marks && {
            symbol: 'none',
            silent: true,
            data: marks.map(([x, name]) => ({ xAxis: x, name })),
            lineStyle: { color: t.muted, type: 'dashed' },
            label: { formatter: '{b}', position: 'start', color: t.muted, fontSize: 11 },
          },
        },
      ],
    }),
    [rows, metric, unit, marks],
  )
  return <Chart option={option} height={(marks ? 40 : 24) + rows.length * 24} ariaLabel={metric === 'fps' ? 'FPS по прогонам' : 'Пиковая RAM по прогонам'} />
}

const RAM_MARKS: [number, string][] = [
  [6, 'цель 6 ГБ'],
  [10, 'предел 10 ГБ'],
]

function ByComponent({ rows }: { rows: Row[] }) {
  const groups = useMemo(() => {
    const m = new Map<string, Row[]>()
    for (const r of rows) {
      const k = `${r.run.mode === 'oracle' ? 'oracle (GT-треклеты)' : r.detector} · ${r.reid}`
      m.set(k, [...(m.get(k) ?? []), r])
    }
    const mean = (xs: (number | null)[]) => {
      const v = xs.filter((x): x is number => x != null)
      return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
    }
    return [...m.entries()].map(([k, rs]) => ({
      k,
      n: rs.length,
      fps: mean(rs.map((r) => r.rt.fps)),
      ram: mean(rs.map((r) => r.rt.peakRssGb)),
      det: mean(rs.map((r) => r.rt.stages.find((s) => s.stage === 'detect')?.p50Ms ?? null)),
      reid: mean(rs.map((r) => r.rt.stages.find((s) => s.stage === 'reid')?.p50Ms ?? null)),
    }))
  }, [rows])
  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/40 text-xs text-muted-foreground">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Детектор · ReID</th>
            <th className="px-3 py-2 text-right font-medium">Прогонов</th>
            <th className="px-3 py-2 text-right font-medium">
              <MetricLabel metric="fps" />
            </th>
            <th className="px-3 py-2 text-right font-medium">
              <MetricLabel metric="peak_rss_gb" />
            </th>
            <th className="px-3 py-2 text-right font-medium">
              <MetricLabel metric="p50_ms" label="детектор p50" />
            </th>
            <th className="px-3 py-2 text-right font-medium">
              <MetricLabel metric="p50_ms" label="ReID p50" />
            </th>
          </tr>
        </thead>
        <tbody className="font-mono text-xs tabular-nums">
          {groups.map((g) => (
            <tr key={g.k} className="border-b last:border-0">
              <td className="px-3 py-1.5">{g.k}</td>
              <td className="px-3 py-1.5 text-right">{g.n}</td>
              <td className="px-3 py-1.5 text-right">{formatMetric('fps', g.fps)}</td>
              <td className="px-3 py-1.5 text-right">{formatMetric('peak_rss_gb', g.ram)}</td>
              <td className="px-3 py-1.5 text-right">{formatMetric('p50_ms', g.det)}</td>
              <td className="px-3 py-1.5 text-right">{formatMetric('p50_ms', g.reid)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Charts({ runs, profiles }: { runs: RunOut[]; profiles: ProfileOut[] }) {
  const metrics = useManyRunMetrics(runs.map((r) => r.id))
  const pending = metrics.some((q) => q.isPending)
  const rows: Row[] = runs.flatMap((run, i) => {
    const data = metrics[i]?.data
    if (!data) return []
    return [
      {
        run,
        rt: parseRuntime(data.runtime),
        label: `#${run.id} ${run.name.length > 28 ? `${run.name.slice(0, 27)}…` : run.name}`,
        detector: componentName(run, profiles, 'detect.name'),
        reid: componentName(run, profiles, 'reid.name'),
      },
    ]
  })
  if (pending && rows.length === 0) return <LoadingBlock rows={6} />
  if (rows.length === 0) return <EmptyState title="Нет данных runtime.json" />
  return (
    <div className="space-y-6">
      <Section title="Время стадий" description="Минуты по стадиям; мелкие стадии сгруппированы">
        <Card className="py-3">
          <CardContent className="px-4">
            <StageStack rows={rows} />
          </CardContent>
        </Card>
      </Section>
      <div className="grid gap-4 xl:grid-cols-2">
        <Section title={<MetricLabel metric="fps" label="FPS (кадры/с суммарно)" />}>
          <Card className="py-3">
            <CardContent className="px-4">
              <RunBars rows={rows} metric="fps" unit="fps" />
            </CardContent>
          </Card>
        </Section>
        <Section title={<MetricLabel metric="peak_rss_gb" label="Пиковая RAM, ГБ" />}>
          <Card className="py-3">
            <CardContent className="px-4">
              <RunBars rows={rows} metric="peak_rss_gb" unit="ГБ" marks={RAM_MARKS} />
            </CardContent>
          </Card>
        </Section>
      </div>
      <Section title="Скорость по компонентам" description="Среднее по выбранным прогонам">
        <ByComponent rows={rows} />
      </Section>
    </div>
  )
}

export function ResourcesPage() {
  const { dataset } = useSearch({ from: '/resources' })
  const navigate = useNavigate({ from: '/resources' })
  const runs = useRuns({ dataset, status: 'done' })
  const datasets = useDatasets()
  const profiles = useProfiles()
  return (
    <>
      <PageHeader
        title="Ресурсы"
        description={`Время стадий, FPS, латентности и пиковая память по последним ${MAX_RUNS} завершённым прогонам`}
        actions={
          <Select value={dataset ?? ALL} onValueChange={(v) => void navigate({ search: { dataset: v === ALL ? undefined : v }, replace: true })}>
            <SelectTrigger size="sm" className="min-w-44" aria-label="Датасет">
              <span className="text-muted-foreground">Датасет:</span>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL}>все</SelectItem>
              {(datasets.data ?? []).map((d) => (
                <SelectItem key={d.id} value={d.id}>
                  {d.title}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        }
      />
      <QueryView
        query={runs}
        loading={<LoadingBlock rows={8} />}
        isEmpty={(rs) => !rs.some((r) => r.status === 'done')}
        empty={
          <EmptyState
            title="Нет завершённых прогонов"
            action={
              <Link to="/runs/new" className="text-sm text-brand hover:underline">
                Запустить прогон
              </Link>
            }
          />
        }
      >
        {(rs) => <Charts runs={rs.filter((r) => r.status === 'done').slice(0, MAX_RUNS)} profiles={profiles.data ?? []} />}
      </QueryView>
    </>
  )
}
