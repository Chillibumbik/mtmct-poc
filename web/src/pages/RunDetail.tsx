import { useCallback } from 'react'
import { Link, useNavigate, useParams } from '@tanstack/react-router'
import { AlertTriangle, ArrowRight, Bug, Download, Film, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import type { EChartsOption } from 'echarts'
import { errorText, urls } from '@/api/client'
import { useDeleteRun, useJobEvents, useRun, useRunErrors, useRunMetrics } from '@/api/hooks'
import type { RunOut } from '@/api/types.gen'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Chart } from '@/components/app/Chart'
import { TransitionMatrix } from '@/features/errors/TransitionMatrix'
import type { ChartTheme } from '@/lib/chart'
import { Confirm } from '@/components/app/ConfirmButton'
import { DiffChips } from '@/components/app/DiffChips'
import { MetricGrid, MetricLabel, MetricTile } from '@/components/app/Metric'
import { PageHeader, Section } from '@/components/app/PageHeader'
import { EmptyState, ErrorState, LoadingBlock, QueryView } from '@/components/app/States'
import { SplitBadge, StatusBadge } from '@/components/app/StatusBadge'
import { getPath } from '@/lib/config'
import { STAGE_LABELS, formatDate, formatEta } from '@/lib/format'
import { METRIC_SECTIONS, formatMetric } from '@/lib/metrics'
import { parseRuntime } from '@/lib/runtime'

type MetricMap = Record<string, number | null>
type Curve = { threshold: number; pairwise_p: number; pairwise_r: number; IDF1: number }[]

function RunLive({ run }: { run: RunOut }) {
  const p = useJobEvents(run.job_id)
  return (
    <Card className="py-4">
      <CardContent className="space-y-2 px-4">
        <div className="flex items-center justify-between text-sm">
          <span>
            {run.status === 'queued' ? 'Прогон в очереди' : `Стадия: ${p.stage ? (STAGE_LABELS[p.stage] ?? p.stage) : '—'}`}
          </span>
          <span className="text-xs tabular-nums text-muted-foreground">
            {Math.round(p.progress * 100)}% · ETA {formatEta(p.etaS)}
          </span>
        </div>
        <Progress value={p.progress * 100} className="h-2" />
        <div className="flex items-center justify-between gap-2">
          <span className="truncate font-mono text-xs text-muted-foreground">{p.logs.at(-1)?.message}</span>
          {run.job_id != null && (
            <Button asChild size="xs" variant="ghost">
              <Link to="/jobs" search={{ job: run.job_id }}>
                Задача #{run.job_id} <ArrowRight />
              </Link>
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function PerCameraTable({ perCamera, mean }: { perCamera: Record<string, MetricMap>; mean?: MetricMap }) {
  const cams = Object.keys(perCamera).sort()
  const keys = METRIC_SECTIONS.per_camera
  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/40 text-xs text-muted-foreground">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Камера</th>
            {keys.map((k) => (
              <th key={k} className="px-2 py-2 text-right font-medium">
                <MetricLabel metric={k} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono text-xs tabular-nums">
          {cams.map((c) => (
            <tr key={c} className="border-b">
              <td className="px-3 py-1.5 text-left">{c}</td>
              {keys.map((k) => (
                <td key={k} className="px-2 py-1.5 text-right">
                  {formatMetric(k, perCamera[c]?.[k])}
                </td>
              ))}
            </tr>
          ))}
          {mean && (
            <tr className="bg-muted/30 font-semibold">
              <td className="px-3 py-1.5 text-left font-sans">среднее</td>
              {keys.map((k) => (
                <td key={k} className="px-2 py-1.5 text-right">
                  {formatMetric(k, mean[k])}
                </td>
              ))}
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

function ThresholdChart({ curve, current }: { curve: Curve; current: number | null }) {
  const option = useCallback(
    (t: ChartTheme): EChartsOption => ({
      tooltip: { trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? (v * 100).toFixed(1) : String(v)) },
      legend: { top: 0, left: 0, data: ['Pairwise P', 'Pairwise R', 'IDF1×'] },
      xAxis: { type: 'value', name: 'порог', nameLocation: 'middle', nameGap: 26, min: 'dataMin', max: 'dataMax' },
      yAxis: { type: 'value', min: 0, max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}` } },
      grid: { left: 8, right: 16, top: 36, bottom: 28, containLabel: true },
      series: [
        { name: 'Pairwise P', key: 'pairwise_p' },
        { name: 'Pairwise R', key: 'pairwise_r' },
        { name: 'IDF1×', key: 'IDF1' },
      ].map((s, i) => ({
        type: 'line' as const,
        name: s.name,
        showSymbol: false,
        lineStyle: { width: 2 },
        data: curve.map((p) => [p.threshold, p[s.key as keyof Curve[number]]]),
        ...(i === 0 && current != null
          ? {
              markLine: {
                symbol: 'none',
                silent: true,
                lineStyle: { color: t.muted, type: 'dashed' as const },
                label: { formatter: `текущий ${current}`, color: t.muted, fontSize: 11 },
                data: [{ xAxis: current }],
              },
            }
          : {}),
      })),
    }),
    [curve, current],
  )
  return <Chart option={option} height={300} ariaLabel="Pairwise precision, recall и межкамерный IDF1 в зависимости от порога решателя" />
}

function StageTimeChart({ stages }: { stages: { stage: string; wallS: number; cached: boolean }[] }) {
  const option = useCallback(
    (t: ChartTheme): EChartsOption => ({
      tooltip: { trigger: 'item', formatter: (p) => `${(p as { name: string }).name}: ${((p as { value: number }).value ?? 0).toFixed(1)} с` },
      grid: { left: 8, right: 40, top: 4, bottom: 4, containLabel: true },
      xAxis: { type: 'value', axisLabel: { formatter: (v: number) => `${v} с` } },
      yAxis: { type: 'category', inverse: true, data: stages.map((s) => `${STAGE_LABELS[s.stage] ?? s.stage}${s.cached ? ' (кеш)' : ''}`), splitLine: { show: false } },
      series: [
        {
          type: 'bar',
          barMaxWidth: 14,
          itemStyle: { color: t.palette[0], borderRadius: [0, 4, 4, 0] },
          label: { show: true, position: 'right', color: t.muted, fontSize: 11, formatter: (p) => `${Number(p.value).toFixed(0)} с` },
          data: stages.map((s) => s.wallS),
        },
      ],
    }),
    [stages],
  )
  return <Chart option={option} height={Math.max(120, stages.length * 26)} ariaLabel="Время стадий прогона" />
}

function Metrics({ run }: { run: RunOut }) {
  const metrics = useRunMetrics(run.id)
  const errors = useRunErrors(run.id)
  return (
    <QueryView query={metrics} loading={<LoadingBlock rows={8} />}>
      {({ metrics: m, runtime }) => {
        const cross = m.cross_camera as MetricMap | undefined
        const identity = (m.identity ?? {}) as MetricMap
        const perCamera = (m.per_camera ?? {}) as Record<string, MetricMap>
        const curve = (m.threshold_curve ?? []) as Curve
        const rt = parseRuntime(runtime)
        const current = getPath(run.config, 'associate.solver.params.threshold') ?? getPath(run.overrides, 'associate.solver.params.threshold') ?? null
        return (
          <div className="space-y-7">
            <Section title="Межкамерные метрики" description="Камеры склеены по оси времени; GT ID — глобальный ID человека (§9.2)">
              <MetricGrid section={cross} keys={METRIC_SECTIONS.cross_camera} />
            </Section>

            <Section title="Идентичности" description="Глобальные ID как кластеризация треклетов (§9.3)">
              <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-6">
                {['pairwise_f1', 'bcubed_f', 'purity', 'inverse_purity', 'handover_success'].map((k) => (
                  <MetricTile key={k} metric={k} value={identity[k]} />
                ))}
                <MetricTile metric="n_pred_ids" value={identity.n_pred_ids} sub={<>GT: {formatMetric('n_gt_ids', identity.n_gt_ids)}</>} />
                {['pairwise_p', 'pairwise_r', 'bcubed_p', 'bcubed_r'].map((k) => (
                  <MetricTile key={k} metric={k} value={identity[k]} />
                ))}
                <MetricTile metric="false_merges" value={identity.false_merges} className={identity.false_merges ? 'border-bad/40' : undefined} />
                <MetricTile metric="false_splits" value={identity.false_splits} className={identity.false_splits ? 'border-bad/40' : undefined} />
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card className="gap-2 py-3">
                  <CardContent className="space-y-2 px-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">
                        <MetricLabel metric="handover_success" label="Handover: камера → камера" />
                      </span>
                      <Button asChild size="xs" variant="ghost">
                        <Link to="/runs/$runId/errors" params={{ runId: String(run.id) }}>
                          <Bug /> Разбор ошибок
                        </Link>
                      </Button>
                    </div>
                    <QueryView query={errors} loading={<LoadingBlock rows={3} />}>
                      {(e) => <TransitionMatrix transitions={e.transitions} />}
                    </QueryView>
                  </CardContent>
                </Card>
                <Card className="gap-2 py-3">
                  <CardContent className="px-4">
                    <div className="mb-1 text-sm font-medium">Кривая по порогу решателя</div>
                    {curve.length ? <ThresholdChart curve={curve} current={typeof current === 'number' ? current : null} /> : <EmptyState title="Кривая не посчитана" className="py-6" />}
                  </CardContent>
                </Card>
              </div>
            </Section>

            <Section title="По камерам" description="TrackEval MotChallenge2DBox, IoU 0.5 (§9.1)">
              {Object.keys(perCamera).length ? (
                <PerCameraTable perCamera={perCamera} mean={m.per_camera_mean as MetricMap | undefined} />
              ) : (
                <EmptyState title="Нет поканальных метрик" className="py-6" />
              )}
            </Section>

            <Section title="ReID-санити" description="Запрос — портрет треклета камеры i, галерея — треклеты других камер (§9.4)">
              <MetricGrid section={m.reid as MetricMap | undefined} keys={METRIC_SECTIONS.reid} className="lg:grid-cols-6" />
            </Section>

            <Section title="Ресурсы" description="§9.6">
              <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
                <div className="grid grid-cols-2 content-start gap-2">
                  <MetricTile metric="wall_time_s" value={rt.wallS} />
                  <MetricTile metric="fps" value={rt.fps} />
                  <MetricTile metric="peak_rss_gb" value={rt.peakRssGb} sub={rt.peakRssGb != null && rt.peakRssGb > 6 ? 'выше цели 6 ГБ' : undefined} />
                  <MetricTile metric="cpu_mean" value={rt.cpuMean} />
                  {rt.stages
                    .filter((s) => s.p50Ms != null)
                    .map((s) => (
                      <MetricTile
                        key={s.stage}
                        metric="p50_ms"
                        label={`${STAGE_LABELS[s.stage] ?? s.stage} p50`}
                        value={s.p50Ms}
                        sub={<>p95 {formatMetric('p95_ms', s.p95Ms)}</>}
                      />
                    ))}
                  <MetricTile metric="cache_mb" value={rt.cacheMb} />
                </div>
                <Card className="gap-2 py-3">
                  <CardContent className="px-4">
                    <div className="mb-1 text-sm font-medium">Время стадий</div>
                    {rt.stages.length ? <StageTimeChart stages={rt.stages} /> : <EmptyState title="runtime.json без стадий" className="py-6" />}
                  </CardContent>
                </Card>
              </div>
            </Section>
          </div>
        )
      }}
    </QueryView>
  )
}

function Config({ run }: { run: RunOut }) {
  return (
    <Section title="Конфигурация" description={`Профиль ${run.profile}; изменения относительно профиля`}>
      <DiffChips diff={run.config_diff} max={20} />
      {run.config && (
        <details className="rounded-lg border">
          <summary className="cursor-pointer px-3 py-2 text-sm text-muted-foreground hover:text-foreground">Полный конфиг (JSON)</summary>
          <pre className="max-h-96 overflow-auto border-t bg-muted/30 p-3 font-mono text-xs">{JSON.stringify(run.config, null, 2)}</pre>
        </details>
      )}
    </Section>
  )
}

export function RunDetailPage() {
  const { runId } = useParams({ from: '/runs/$runId' })
  const id = Number(runId)
  const run = useRun(id)
  const del = useDeleteRun()
  const navigate = useNavigate()
  if (!Number.isInteger(id)) return <EmptyState title="Некорректный номер прогона" />
  if (run.isPending) return <LoadingBlock rows={10} />
  if (run.isError) return <ErrorState error={run.error} onRetry={() => void run.refetch()} />
  const r = run.data
  const done = r.status === 'done'
  return (
    <div className="space-y-6">
      <PageHeader
        title={r.name}
        description={
          <span className="flex flex-wrap items-center gap-1.5">
            <span className="font-mono">#{r.id}</span>
            <StatusBadge status={r.status} />
            <span>
              {r.dataset}/<span className="font-mono">{r.sequence}</span>
            </span>
            <SplitBadge split={r.split} />
            <Badge variant="outline" className="font-mono font-normal">
              {r.mode}
            </Badge>
            <Badge variant="secondary" className="font-normal">
              {r.profile}
            </Badge>
            <span>
              · {formatDate(r.created_at)}
              {r.finished_at && ` → ${formatDate(r.finished_at)}`}
            </span>
            {r.git_sha && <span className="font-mono">· {r.git_sha.slice(0, 7)}</span>}
            {r.sweep_id != null && (
              <Link to="/experiments" search={{ sweep: r.sweep_id }} className="hover:underline">
                · sweep #{r.sweep_id}
              </Link>
            )}
          </span>
        }
        actions={
          <>
            <Button asChild size="sm" variant="outline" disabled={!done}>
              <Link to="/runs/$runId/player" params={{ runId }}>
                <Film /> Плеер
              </Link>
            </Button>
            <Button asChild size="sm" variant="outline" disabled={!done}>
              <Link to="/runs/$runId/errors" params={{ runId }}>
                <Bug /> Ошибки
              </Link>
            </Button>
            {done && (
              <Button asChild size="sm" variant="outline">
                <a href={urls.export(r.id)} download>
                  <Download /> Экспорт
                </a>
              </Button>
            )}
            <Confirm
              trigger={
                <Button size="sm" variant="ghost" aria-label="Удалить прогон" disabled={r.status === 'running' || r.status === 'queued'}>
                  <Trash2 />
                </Button>
              }
              title={`Удалить прогон #${r.id}?`}
              description="Будут удалены результаты прогона (треки, метрики, логи). Кеш тяжёлых стадий сохранится."
              confirmLabel="Удалить"
              onConfirm={() =>
                del.mutate(r.id, {
                  onSuccess: () => {
                    toast.success(`Прогон #${r.id} удалён`)
                    void navigate({ to: '/experiments' })
                  },
                  onError: (e) => toast.error(errorText(e)),
                })
              }
            />
          </>
        }
      />
      {(r.status === 'running' || r.status === 'queued') && <RunLive run={r} />}
      {(r.status === 'failed' || r.status === 'cancelled') && (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm">
          <AlertTriangle className="mt-0.5 size-4 text-bad" />
          <div className="flex-1">
            <div className="font-medium">{r.status === 'failed' ? 'Прогон завершился с ошибкой' : 'Прогон отменён'}</div>
            {r.error && <div className="text-muted-foreground">{r.error}</div>}
          </div>
          {r.job_id != null && (
            <Button asChild size="sm" variant="outline">
              <Link to="/jobs" search={{ job: r.job_id }}>
                Задача и лог
              </Link>
            </Button>
          )}
        </div>
      )}
      {done && <Metrics run={r} />}
      <Config run={r} />
    </div>
  )
}
