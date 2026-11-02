import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useSearch } from '@tanstack/react-router'
import { ArrowRight, Ban, CheckCircle2, Circle, CircleDashed, Loader2, RotateCw, SkipForward, XCircle } from 'lucide-react'
import { toast } from 'sonner'
import { errorText } from '@/api/client'
import { useCancelJob, useJob, useJobEvents, useJobs, useRestartJob } from '@/api/hooks'
import type { JobEvent, JobOut } from '@/api/types.gen'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Confirm } from '@/components/app/ConfirmButton'
import { PageHeader } from '@/components/app/PageHeader'
import { EmptyState, ErrorState, LoadingBlock, QueryView } from '@/components/app/States'
import { StatusBadge } from '@/components/app/StatusBadge'
import { JOB_KIND_LABELS, STAGE_LABELS, formatDate, formatEta, formatRelative } from '@/lib/format'
import { isTerminal, stageList, type StageState } from '@/lib/jobProgress'
import { cn } from '@/lib/utils'

const STAGE_ICON: Record<StageState, { icon: typeof Circle; cls: string }> = {
  pending: { icon: Circle, cls: 'text-muted-foreground/50' },
  running: { icon: Loader2, cls: 'animate-spin text-brand' },
  done: { icon: CheckCircle2, cls: 'text-good' },
  failed: { icon: XCircle, cls: 'text-bad' },
  skipped: { icon: SkipForward, cls: 'text-muted-foreground/60' },
}

const LEVEL_CLS: Record<JobEvent['level'], string> = {
  debug: 'text-muted-foreground/70',
  info: '',
  warning: 'text-warn',
  error: 'text-bad',
}

type LevelFilter = 'all' | 'info' | 'warning'

function Logs({ logs }: { logs: JobEvent[] }) {
  const [level, setLevel] = useState<LevelFilter>('info')
  const [follow, setFollow] = useState(true)
  const ref = useRef<HTMLDivElement>(null)
  const shown = useMemo(
    () => logs.filter((l) => level === 'all' || (level === 'info' ? l.level !== 'debug' : l.level === 'warning' || l.level === 'error')),
    [logs, level],
  )
  useEffect(() => {
    if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [shown, follow])
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium">Лог</span>
        <ToggleGroup type="single" size="sm" variant="outline" value={level} onValueChange={(v) => v && setLevel(v as LevelFilter)}>
          <ToggleGroupItem value="all" className="px-2 text-xs">
            все
          </ToggleGroupItem>
          <ToggleGroupItem value="info" className="px-2 text-xs">
            info+
          </ToggleGroupItem>
          <ToggleGroupItem value="warning" className="px-2 text-xs">
            предупр.
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      <div
        ref={ref}
        onScroll={(e) => {
          const el = e.currentTarget
          setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24)
        }}
        className="h-72 overflow-auto rounded-md border bg-muted/30 p-2 font-mono text-[11.5px] leading-5"
        role="log"
        aria-live="polite"
      >
        {shown.length === 0 && <div className="text-muted-foreground">Событий пока нет</div>}
        {shown.map((l) => (
          <div key={l.id} className={cn('flex gap-2 whitespace-pre-wrap', LEVEL_CLS[l.level])}>
            <span className="shrink-0 text-muted-foreground/70 tabular-nums">{new Date(l.ts).toLocaleTimeString('ru-RU')}</span>
            {l.stage && <span className="w-20 shrink-0 truncate text-muted-foreground">{l.stage}</span>}
            <span className="min-w-0 break-words">{l.message}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function JobActions({ job }: { job: JobOut }) {
  const cancel = useCancelJob()
  const restart = useRestartJob()
  const navigate = useNavigate({ from: '/jobs' })
  return (
    <div className="flex flex-wrap gap-2">
      {(job.status === 'queued' || job.status === 'running') && (
        <Confirm
          trigger={
            <Button size="sm" variant="outline" disabled={cancel.isPending}>
              <Ban /> Отменить
            </Button>
          }
          title={`Отменить задачу #${job.id}?`}
          description="Текущая стадия будет прервана; готовые стадии остаются в кеше и подхватятся при перезапуске."
          confirmLabel="Отменить задачу"
          onConfirm={() => cancel.mutate(job.id, { onError: (e) => toast.error(errorText(e)) })}
        />
      )}
      {(job.status === 'failed' || job.status === 'cancelled') && job.kind !== 'sweep' && (
        <Button
          size="sm"
          variant="outline"
          disabled={restart.isPending}
          onClick={() =>
            restart.mutate(job.id, {
              onSuccess: (j) => {
                toast.success(`Перезапущено: задача #${j.id}`)
                void navigate({ search: { job: j.id } })
              },
              onError: (e) => toast.error(errorText(e)),
            })
          }
        >
          <RotateCw /> Перезапустить
        </Button>
      )}
      {job.run_id != null && (
        <Button asChild size="sm" variant={job.status === 'done' ? 'default' : 'ghost'}>
          <Link to="/runs/$runId" params={{ runId: String(job.run_id) }}>
            Прогон #{job.run_id} <ArrowRight />
          </Link>
        </Button>
      )}
      {job.dataset_id && (
        <Button asChild size="sm" variant="ghost">
          <Link to="/data">Данные</Link>
        </Button>
      )}
    </div>
  )
}

function JobDetail({ jobId }: { jobId: number }) {
  const job = useJob(jobId)
  const live = useJobEvents(jobId)
  if (job.isPending) return <LoadingBlock rows={6} />
  if (job.isError) return <ErrorState error={job.error} onRetry={() => void job.refetch()} />
  const j = job.data
  // The stream is fresher than the polled job row while running; the row wins once it is terminal.
  const status = isTerminal(j.status) ? j.status : (live.status ?? j.status)
  const progress = status === 'done' ? 1 : Math.max(live.progress, j.progress)
  const stages = stageList(live, j.kind)
  return (
    <Card className="gap-4 py-4">
      <CardHeader className="space-y-1 px-4">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="text-sm">
            #{j.id} · {j.title}
          </CardTitle>
          <StatusBadge status={status} />
          <Badge variant="secondary" className="font-normal">
            {JOB_KIND_LABELS[j.kind] ?? j.kind}
          </Badge>
          {live.conn === 'connecting' && !isTerminal(status) && (
            <span className="flex items-center gap-1 text-xs text-muted-foreground">
              <CircleDashed className="size-3 animate-spin" /> подключение…
            </span>
          )}
        </div>
        <div className="text-xs text-muted-foreground">
          создана {formatDate(j.created_at)} · старт {formatDate(j.started_at)} · конец {formatDate(j.finished_at)}
        </div>
      </CardHeader>
      <CardContent className="space-y-4 px-4">
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span>{live.stage || j.stage ? (STAGE_LABELS[live.stage ?? j.stage ?? ''] ?? live.stage ?? j.stage) : '—'}</span>
            <span className="tabular-nums text-muted-foreground">
              {Math.round(progress * 100)}%{!isTerminal(status) && ` · ETA ${formatEta(live.etaS ?? j.eta_s)}`}
            </span>
          </div>
          <Progress value={progress * 100} className="h-2" />
        </div>
        {(j.error || status === 'failed') && (
          <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-bad">{j.error ?? 'Задача завершилась с ошибкой'}</div>
        )}
        {stages.length > 0 && (
          <ol className="gap-x-6 space-y-1.5 sm:columns-2">
            {stages.map((s) => {
              const ic = STAGE_ICON[s.state]
              const Icon = ic.icon
              return (
                <li key={s.name} className="flex break-inside-avoid items-center gap-2 text-xs">
                  <Icon className={cn('size-3.5 shrink-0', ic.cls)} />
                  <span className={cn('w-32 truncate', s.state === 'pending' || s.state === 'skipped' ? 'text-muted-foreground' : '')}>
                    {STAGE_LABELS[s.name] ?? s.name}
                  </span>
                  <Progress value={s.progress * 100} className={cn('h-1 flex-1', s.state === 'skipped' && 'opacity-30')} />
                  <span className="w-9 text-right tabular-nums text-muted-foreground">{s.state === 'skipped' ? '—' : `${Math.round(s.progress * 100)}%`}</span>
                </li>
              )
            })}
          </ol>
        )}
        <JobActions job={{ ...j, status }} />
        <Logs logs={live.logs} />
      </CardContent>
    </Card>
  )
}

export function JobsPage() {
  const { job: selected } = useSearch({ from: '/jobs' })
  const navigate = useNavigate({ from: '/jobs' })
  const jobs = useJobs()
  const active = selected ?? jobs.data?.find((j) => j.status === 'running')?.id ?? jobs.data?.[0]?.id
  return (
    <>
      <PageHeader title="Задачи" description="Очередь воркера: одна тяжёлая задача за раз; прогресс приходит потоком SSE" />
      <QueryView
        query={jobs}
        loading={<LoadingBlock rows={8} />}
        isEmpty={(js) => js.length === 0}
        empty={<EmptyState title="Задач пока нет" hint="Запустите прогон или скачайте датасет." />}
      >
        {(js) => (
          <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="overflow-hidden rounded-lg border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-14">#</TableHead>
                    <TableHead>Задача</TableHead>
                    <TableHead className="w-28">Статус</TableHead>
                    <TableHead className="w-32">Прогресс</TableHead>
                    <TableHead className="w-24 text-right">Создана</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {js.map((j) => (
                    <TableRow
                      key={j.id}
                      data-state={j.id === active ? 'selected' : undefined}
                      className="cursor-pointer"
                      onClick={() => void navigate({ search: { job: j.id }, replace: true })}
                    >
                      <TableCell className="font-mono text-xs text-muted-foreground">{j.id}</TableCell>
                      <TableCell className="max-w-0">
                        <div className="truncate font-medium">{j.title}</div>
                        <div className="truncate text-xs text-muted-foreground">
                          {JOB_KIND_LABELS[j.kind] ?? j.kind}
                          {j.stage && j.status === 'running' && ` · ${STAGE_LABELS[j.stage] ?? j.stage}`}
                          {j.error && ` · ${j.error}`}
                        </div>
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={j.status} />
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Progress value={j.progress * 100} className="h-1.5" />
                          <span className="w-8 text-right text-xs tabular-nums text-muted-foreground">{Math.round(j.progress * 100)}%</span>
                        </div>
                      </TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground">{formatRelative(j.created_at)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            {active != null ? <JobDetail key={active} jobId={active} /> : null}
          </div>
        )}
      </QueryView>
    </>
  )
}
