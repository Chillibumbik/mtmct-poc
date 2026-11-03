import { useCallback } from 'react'
import { Link, useNavigate, useSearch } from '@tanstack/react-router'
import { ArrowLeftToLine, GitCompare, X } from 'lucide-react'
import type { EChartsOption } from 'echarts'
import { useCompare, useManyRunMetrics } from '@/api/hooks'
import type { CompareOut, RunOut } from '@/api/types.gen'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Chart } from '@/components/app/Chart'
import { DiffChips } from '@/components/app/DiffChips'
import { MetricLabel } from '@/components/app/Metric'
import { PageHeader, Section } from '@/components/app/PageHeader'
import { EmptyState, LoadingBlock, QueryView } from '@/components/app/States'
import { SplitBadge } from '@/components/app/StatusBadge'
import type { ChartTheme } from '@/lib/chart'
import { formatMetric } from '@/lib/metrics'
import { cn } from '@/lib/utils'
import { bestRuns, deltaTone, formatDelta, runLabel, varyingKeys } from './compare'

type Curve = { threshold: number; pairwise_p: number; pairwise_r: number; IDF1: number }[]

const pct = (v: unknown) => (typeof v === 'number' ? (v * 100).toFixed(1) : String(v))

function AblationChart({ data }: { data: CompareOut }) {
  const option = useCallback(
    (t: ChartTheme): EChartsOption => {
      const row = (m: string) => data.rows.find((r) => r.metric === m)
      const series = [
        { name: 'HOTA×', metric: 'cross_hota' },
        { name: 'IDF1×', metric: 'cross_idf1' },
      ]
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: pct },
        legend: { top: 0, left: 0, data: series.map((s) => s.name) },
        grid: { left: 8, right: 16, top: 36, bottom: 8, containLabel: true },
        // axis shows `#id`, the tooltip the full label
        xAxis: { type: 'category', data: data.runs.map(runLabel), axisLabel: { color: t.muted, fontSize: 11, interval: 0, formatter: (v: string) => v.split(' ')[0] } },
        yAxis: { type: 'value', min: 0, max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}` } },
        series: series.map((s, i) => ({
          type: 'bar' as const,
          name: s.name,
          barMaxWidth: 28,
          itemStyle: { color: t.palette[i], borderRadius: [3, 3, 0, 0] },
          data: data.runs.map((r) => row(s.metric)?.values[r.id] ?? null),
        })),
      }
    },
    [data],
  )
  return <Chart option={option} height={240} ariaLabel="Межкамерные HOTA и IDF1 по прогонам" />
}

// legend entries per row at half page width (P/R entries are longer)
const legendRowsFor = (kind: 'idf1' | 'pr', nRuns: number) => (kind === 'idf1' ? Math.ceil(nRuns / 3) : nRuns)

function CurveChart({ runs, curves, kind }: { runs: RunOut[]; curves: (Curve | null)[]; kind: 'idf1' | 'pr' }) {
  const option = useCallback(
    (t: ChartTheme): EChartsOption => {
      const series = runs.flatMap((r, i) => {
        const c = curves[i]
        if (!c?.length) return []
        const color = t.palette[i % t.palette.length]
        const line = (name: string, key: keyof Curve[number], dashed = false) => ({
          type: 'line' as const,
          name,
          showSymbol: false,
          lineStyle: { width: 2, color, type: dashed ? ('dashed' as const) : ('solid' as const) },
          itemStyle: { color },
          data: c.map((p) => [p.threshold, p[key]]),
        })
        return kind === 'idf1' ? [line(runLabel(r), 'IDF1')] : [line(`${runLabel(r)} · P`, 'pairwise_p'), line(`${runLabel(r)} · R`, 'pairwise_r', true)]
      })
      const legendRows = legendRowsFor(kind, runs.length)
      return {
        tooltip: { trigger: 'axis', valueFormatter: pct },
        legend: { top: 0, left: 0 },
        grid: { left: 8, right: 16, top: 16 + 22 * legendRows, bottom: 28, containLabel: true },
        xAxis: { type: 'value', name: 'порог решателя', nameLocation: 'middle', nameGap: 26, min: 'dataMin', max: 'dataMax' },
        yAxis: { type: 'value', min: 0, max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}` } },
        series,
      }
    },
    [runs, curves, kind],
  )
  return (
    <Chart
      option={option}
      height={280 + 22 * legendRowsFor(kind, runs.length)}
      ariaLabel={kind === 'idf1' ? 'Межкамерный IDF1 в зависимости от порога, по прогонам' : 'Pairwise precision (сплошная) и recall (пунктир) по порогу, по прогонам'}
    />
  )
}

function CompareTable({ data, onBase, onRemove }: { data: CompareOut; onBase: (id: number) => void; onRemove: (id: number) => void }) {
  const runs = data.runs
  const varying = varyingKeys(runs)
  const rows = data.rows.filter((r) => runs.some((run) => r.values[run.id] != null))
  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/40 align-top text-xs">
          <tr>
            <th className="sticky left-0 z-[1] min-w-36 bg-muted/40 px-3 py-2 text-left font-medium text-muted-foreground backdrop-blur">Метрика</th>
            {runs.map((r, i) => (
              <th key={r.id} className="min-w-44 px-3 py-2 text-left font-normal">
                <div className="flex items-start justify-between gap-1">
                  <Link to="/runs/$runId" params={{ runId: String(r.id) }} className="font-medium hover:underline">
                    {r.name}
                  </Link>
                  <div className="flex shrink-0">
                    {i > 0 && (
                      <Button size="icon-xs" variant="ghost" onClick={() => onBase(r.id)} title="Сделать базой для дельт" aria-label={`Сделать прогон #${r.id} базой`}>
                        <ArrowLeftToLine />
                      </Button>
                    )}
                    {runs.length > 1 && (
                      <Button size="icon-xs" variant="ghost" onClick={() => onRemove(r.id)} title="Убрать из сравнения" aria-label={`Убрать прогон #${r.id}`}>
                        <X />
                      </Button>
                    )}
                  </div>
                </div>
                <div className="mt-0.5 flex flex-wrap items-center gap-1 text-muted-foreground">
                  <span className="font-mono">#{r.id}</span>
                  {i === 0 && <Badge className="h-4 px-1 text-[10px]">база</Badge>}
                  <span>
                    {r.dataset}/<span className="font-mono">{r.sequence}</span>
                  </span>
                  <SplitBadge split={r.split} />
                  <span className="font-mono">{r.mode}</span>
                </div>
                <div className="mt-1.5">
                  {varying.length ? (
                    <DiffChips diff={Object.fromEntries(varying.map((k) => [k, r.config_diff[k] ?? 'по умолч.']))} max={4} />
                  ) : (
                    <span className="text-muted-foreground">конфиг как у остальных</span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const best = bestRuns(row)
            return (
              <tr key={row.metric} className="border-b last:border-0">
                <td className="sticky left-0 z-[1] bg-background px-3 py-1.5 text-xs">
                  <MetricLabel metric={row.metric} />
                  <span className="ml-1 text-muted-foreground">{row.higher_is_better ? '↑' : '↓'}</span>
                </td>
                {runs.map((r, i) => {
                  const d = row.deltas[r.id]
                  const tone = deltaTone(d, row.higher_is_better)
                  return (
                    <td key={r.id} className={cn('px-3 py-1.5 font-mono text-xs tabular-nums', best.has(r.id) && 'bg-good/10')}>
                      <span className={cn(best.has(r.id) && 'font-semibold')}>{formatMetric(row.metric, row.values[r.id])}</span>
                      {best.has(r.id) && <span className="ml-1 text-good" aria-label="лучший">★</span>}
                      {i > 0 && d != null && (
                        <span className={cn('ml-2', tone === 'good' ? 'text-good' : tone === 'bad' ? 'text-bad' : 'text-muted-foreground')}>
                          {formatDelta(row.metric, d)}
                        </span>
                      )}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** `/compare?runs=a,b,c` — key metrics with deltas vs the first run, ablation bars, threshold curves. */
export function ComparePage() {
  const search = useSearch({ from: '/compare' })
  const navigate = useNavigate({ from: '/compare' })
  const ids = [...new Set((search.runs ?? '').split(',').map(Number).filter((n) => Number.isInteger(n) && n > 0))]
  const q = useCompare(ids)
  const metrics = useManyRunMetrics(ids)
  const setIds = (next: number[]) => void navigate({ search: { runs: next.join(',') } })

  if (!ids.length)
    return (
      <>
        <PageHeader title="Сравнение прогонов" />
        <EmptyState
          icon={<GitCompare className="size-6" />}
          title="Прогоны не выбраны"
          hint="Отметьте 2 и более прогонов в таблице экспериментов и нажмите «Сравнить»."
          action={
            <Button asChild size="sm" variant="outline">
              <Link to="/experiments">К экспериментам</Link>
            </Button>
          }
        />
      </>
    )

  const curves = metrics.map((m) => (m.data?.metrics.threshold_curve as Curve | undefined) ?? null)
  return (
    <div className="space-y-6">
      <PageHeader
        title="Сравнение прогонов"
        description={`${ids.length} ${ids.length === 1 ? 'прогон' : 'прогонов'}; дельты — относительно первого (базового), ★ — лучшее значение строки`}
        actions={
          <Button asChild size="sm" variant="outline">
            <Link to="/experiments">К экспериментам</Link>
          </Button>
        }
      />
      <QueryView query={q} loading={<LoadingBlock rows={8} />}>
        {(data) => {
          const common = Object.fromEntries(
            Object.entries(data.runs[0]?.config_diff ?? {}).filter(([k]) => !varyingKeys(data.runs).includes(k) && data.runs.every((r) => k in r.config_diff)),
          )
          const ready = data.runs.filter((r) => r.status === 'done').length
          return (
            <>
              {ready < data.runs.length && (
                <div className="rounded-lg border px-3 py-2 text-xs text-muted-foreground">
                  {data.runs.length - ready} из {data.runs.length} прогонов ещё не завершены — их метрики пусты.
                </div>
              )}
              {Object.keys(common).length > 0 && (
                <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  Общие изменения конфига: <DiffChips diff={common} max={6} />
                </div>
              )}
              <CompareTable
                data={data}
                onBase={(id) => setIds([id, ...ids.filter((x) => x !== id)])}
                onRemove={(id) => setIds(ids.filter((x) => x !== id))}
              />
              <div className="grid gap-4 lg:grid-cols-2">
                <div className="lg:col-span-2">
                  <Section title="Абляция: межкамерные HOTA и IDF1" description="Вклад изменений конфигурации по прогонам (подписи — номера прогонов); ↑ лучше">
                    <Card className="py-3">
                      <CardContent className="px-4">
                        <AblationChart data={data} />
                      </CardContent>
                    </Card>
                  </Section>
                </div>
                <Section title="IDF1× по порогу решателя" description="Дешёвый перебор только ассоциации (§9.3)">
                  <Card className="py-3">
                    <CardContent className="px-4">
                      {curves.some((c) => c?.length) ? <CurveChart runs={data.runs} curves={curves} kind="idf1" /> : <EmptyState title="Кривые не посчитаны" className="py-8" />}
                    </CardContent>
                  </Card>
                </Section>
                <Section title="Pairwise P / R по порогу" description="Сплошная — precision, пунктир — recall">
                  <Card className="py-3">
                    <CardContent className="px-4">
                      {curves.some((c) => c?.length) ? <CurveChart runs={data.runs} curves={curves} kind="pr" /> : <EmptyState title="Кривые не посчитаны" className="py-8" />}
                    </CardContent>
                  </Card>
                </Section>
              </div>
            </>
          )
        }}
      </QueryView>
    </div>
  )
}
