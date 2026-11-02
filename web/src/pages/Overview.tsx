import { Link } from '@tanstack/react-router'
import { ArrowRight, Cpu, HardDrive, MemoryStick, PlayCircle } from 'lucide-react'
import { useDatasets, useJobs, useRuns, useSystem } from '@/api/hooks'
import type { RunOut } from '@/api/types.gen'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { MetricTile } from '@/components/app/Metric'
import { PageHeader, Section } from '@/components/app/PageHeader'
import { EmptyState, LoadingBlock, QueryView } from '@/components/app/States'
import { OverlapBadge, StatusBadge } from '@/components/app/StatusBadge'
import { JOB_KIND_LABELS, STAGE_LABELS, formatEta, formatRelative } from '@/lib/format'

/** Best finished run per dataset: full pipeline before oracle, then cross-camera HOTA, then IDF1. */
function bestRunPerDataset(runs: RunOut[]): Map<string, RunOut> {
  const best = new Map<string, RunOut>()
  const score = (r: RunOut) => [(r.mode === 'full' ? 2 : 0) + (r.key_metrics.cross_hota ?? -1), r.key_metrics.cross_idf1 ?? -1]
  for (const r of runs) {
    if (r.status !== 'done') continue
    const cur = best.get(r.dataset)
    const [a, b] = score(r)
    const [c, d] = cur ? score(cur) : [-2, -2]
    if (a > c || (a === c && b > d)) best.set(r.dataset, r)
  }
  return best
}

function BestRuns() {
  const runs = useRuns({ split: 'test' })
  const datasets = useDatasets()
  return (
    <QueryView
      query={runs}
      loading={<LoadingBlock rows={4} />}
      isEmpty={(rs) => !rs.some((r) => r.status === 'done')}
      empty={
        <EmptyState
          title="Нет завершённых прогонов на test-сплите"
          hint="Итоговые цифры считаются только на test; параметры подбираются на tune."
          action={
            <Button asChild size="sm">
              <Link to="/runs/new">
                <PlayCircle /> Новый прогон
              </Link>
            </Button>
          }
        />
      }
    >
      {(rs) => (
        <div className="grid gap-3 xl:grid-cols-2">
          {[...bestRunPerDataset(rs).values()].map((r) => {
            const ds = datasets.data?.find((d) => d.id === r.dataset)
            const k = r.key_metrics
            return (
              <Card key={r.dataset} className="gap-3 py-4">
                <CardHeader className="px-4">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <CardTitle className="flex items-center gap-2 text-sm">
                        {ds?.title ?? r.dataset}
                        {ds && <OverlapBadge overlap={ds.overlap} />}
                        <Badge variant="outline" className="font-mono font-normal">
                          {r.mode}
                        </Badge>
                      </CardTitle>
                      <Link
                        to="/runs/$runId"
                        params={{ runId: String(r.id) }}
                        className="mt-1 block truncate text-xs text-muted-foreground hover:text-foreground hover:underline"
                      >
                        #{r.id} · {r.name} · {r.sequence}
                      </Link>
                    </div>
                    <span className="shrink-0 text-xs text-muted-foreground">{formatRelative(r.finished_at)}</span>
                  </div>
                </CardHeader>
                <CardContent className="grid grid-cols-2 gap-2 px-4 sm:grid-cols-5">
                  <MetricTile metric="cross_hota" value={k.cross_hota} label="HOTA×" />
                  <MetricTile metric="cross_idf1" value={k.cross_idf1} label="IDF1×" />
                  <MetricTile metric="handover_success" value={k.handover_success} />
                  <MetricTile metric="fps" value={k.fps} />
                  <MetricTile metric="peak_rss_gb" value={k.peak_rss_gb} />
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}
    </QueryView>
  )
}

function Queue() {
  const jobs = useJobs()
  return (
    <Card className="gap-3 py-4">
      <CardHeader className="flex flex-row items-center justify-between px-4">
        <CardTitle className="text-sm">Очередь задач</CardTitle>
        <Button asChild variant="ghost" size="xs">
          <Link to="/jobs">
            Все задачи <ArrowRight />
          </Link>
        </Button>
      </CardHeader>
      <CardContent className="px-4">
        <QueryView query={jobs}>
          {(js) => {
            const running = js.filter((j) => j.status === 'running')
            const queued = js.filter((j) => j.status === 'queued')
            const failed = js.filter((j) => j.status === 'failed').slice(0, 3)
            return (
              <div className="space-y-3 text-sm">
                <div className="flex gap-4 text-xs text-muted-foreground">
                  <span>
                    выполняется <b className="text-foreground tabular-nums">{running.length}</b>
                  </span>
                  <span>
                    в очереди <b className="text-foreground tabular-nums">{queued.length}</b>
                  </span>
                  <span>
                    с ошибкой <b className="text-foreground tabular-nums">{js.filter((j) => j.status === 'failed').length}</b>
                  </span>
                </div>
                {running.length === 0 && <div className="text-muted-foreground">Воркер простаивает.</div>}
                {running.map((j) => (
                  <Link key={j.id} to="/jobs" search={{ job: j.id }} className="block space-y-1.5 rounded-md border p-2.5 hover:bg-muted/50">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate font-medium">{j.title}</span>
                      <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                        {Math.round(j.progress * 100)}% · ETA {formatEta(j.eta_s)}
                      </span>
                    </div>
                    <Progress value={j.progress * 100} className="h-1.5" />
                    <div className="text-xs text-muted-foreground">
                      {JOB_KIND_LABELS[j.kind]} · стадия: {j.stage ? (STAGE_LABELS[j.stage] ?? j.stage) : '—'}
                    </div>
                  </Link>
                ))}
                {failed.map((j) => (
                  <Link key={j.id} to="/jobs" search={{ job: j.id }} className="flex items-center justify-between gap-2 text-xs hover:underline">
                    <span className="truncate">{j.title}</span>
                    <StatusBadge status={j.status} />
                  </Link>
                ))}
              </div>
            )
          }}
        </QueryView>
      </CardContent>
    </Card>
  )
}

function SystemCard() {
  const sys = useSystem()
  return (
    <Card className="gap-3 py-4">
      <CardHeader className="px-4">
        <CardTitle className="text-sm">Система</CardTitle>
      </CardHeader>
      <CardContent className="px-4">
        <QueryView query={sys}>
          {(s) => {
            const usedRam = 1 - s.ram_available_gb / s.ram_total_gb
            return (
              <div className="space-y-3 text-sm">
                <div className="flex items-center gap-2">
                  <Cpu className="size-4 text-muted-foreground" />
                  <span className="truncate">{s.cpu}</span>
                  <span className="ml-auto shrink-0 text-xs text-muted-foreground tabular-nums">{s.cpu_threads} потоков</span>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <MemoryStick className="size-4 text-muted-foreground" />
                    RAM
                    <span className="ml-auto text-xs tabular-nums text-muted-foreground">
                      свободно {s.ram_available_gb.toFixed(1)} из {s.ram_total_gb.toFixed(1)} ГБ
                    </span>
                  </div>
                  <Progress value={usedRam * 100} className="h-1.5" />
                </div>
                <div className="flex items-center gap-2">
                  <HardDrive className="size-4 text-muted-foreground" />
                  Диск
                  <span className="ml-auto text-xs tabular-nums text-muted-foreground">свободно {s.disk_free_gb.toFixed(0)} ГБ</span>
                </div>
                <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                  OpenVINO:
                  {s.openvino_devices.map((d) => (
                    <Badge key={d} variant="outline" className="font-mono font-normal">
                      {d}
                    </Badge>
                  ))}
                  <span className="ml-auto font-mono">
                    v{s.version}
                    {s.git_sha ? ` · ${s.git_sha.slice(0, 7)}` : ''}
                  </span>
                </div>
              </div>
            )
          }}
        </QueryView>
      </CardContent>
    </Card>
  )
}

function Licences() {
  const sys = useSystem()
  if (!sys.data?.licences.length) return null
  return (
    <details className="group rounded-lg border">
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium">
        Лицензии <span className="font-normal text-muted-foreground">· модели, библиотеки и датасеты ({sys.data.licences.length})</span>
      </summary>
      <div className="overflow-hidden border-t">
        <table className="w-full text-sm">
          <tbody>
            {sys.data.licences.map((l) => (
              <tr key={`${l.kind}-${l.name}`} className="border-b last:border-0">
                <td className="w-24 px-3 py-1.5 text-xs text-muted-foreground">
                  {{ model: 'модель', dataset: 'датасет', library: 'библиотека' }[l.kind]}
                </td>
                <td className="px-3 py-1.5">{l.name}</td>
                <td className="px-3 py-1.5 font-mono text-xs">{l.licence}</td>
                <td className="px-3 py-1.5 text-xs text-muted-foreground">{l.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}

export function OverviewPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Обзор"
        description="Лучший прогон на каждом датасете (test-сплит), состояние очереди и системы"
        actions={
          <Button asChild size="sm">
            <Link to="/runs/new">
              <PlayCircle /> Новый прогон
            </Link>
          </Button>
        }
      />
      <Section title="Лучшие результаты · test" description="Межкамерные HOTA / IDF1 (×), доля успешных handover, скорость и пиковая память">
        <BestRuns />
      </Section>
      <div className="grid gap-3 lg:grid-cols-2">
        <Queue />
        <SystemCard />
      </div>
      <Licences />
    </div>
  )
}
