import { useState } from 'react'
import { Link, useNavigate, useParams, useSearch } from '@tanstack/react-router'
import { ArrowLeft, Film, Play } from 'lucide-react'
import { useRun, useRunErrors } from '@/api/hooks'
import type { ErrorCase, RunOut, TrackletCrop } from '@/api/types.gen'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { MetricLabel, MetricTile } from '@/components/app/Metric'
import { PageHeader, Section } from '@/components/app/PageHeader'
import { EmptyState, ErrorState, LoadingBlock, QueryView } from '@/components/app/States'
import { IdChip } from '@/features/player/PersonPanel'
import { formatClock } from '@/features/player/format'
import { TransitionMatrix } from './TransitionMatrix'

const PAGE = 20

function PlayerLink({ runId, t, gid, camera, label }: { runId: number; t: number; gid?: number; camera?: string; label?: string }) {
  return (
    <Button asChild size="xs" variant={label ? 'outline' : 'ghost'}>
      <Link to="/runs/$runId/player" params={{ runId: String(runId) }} search={{ t: Math.round(t * 100) / 100, gid, camera }} title="Открыть в плеере в этот момент">
        <Play /> {label}
      </Link>
    </Button>
  )
}

function Tracklet({ runId, tr }: { runId: number; tr: TrackletCrop }) {
  return (
    <div className="flex gap-2 rounded-md border p-1.5">
      <div className="flex shrink-0 gap-0.5">
        {tr.crop_urls.map((u) => (
          <img
            key={u}
            src={u}
            alt={`треклет ${tr.tracklet_id}`}
            loading="lazy"
            className="h-24 w-auto max-w-16 rounded-sm bg-muted object-cover"
            onError={(e) => (e.currentTarget.style.visibility = 'hidden')}
          />
        ))}
      </div>
      <div className="min-w-0 flex-1 space-y-0.5 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="font-mono font-medium">{tr.camera_id}</span>
          <IdChip gid={tr.global_id} />
          <PlayerLink runId={runId} t={tr.t_start} gid={tr.global_id} camera={tr.camera_id} />
        </div>
        <div className="tabular-nums text-muted-foreground">
          {formatClock(tr.t_start)}–{formatClock(tr.t_end)}
        </div>
        <div className="text-muted-foreground">
          GT: <span className="font-mono">{tr.gt_person_id ?? '—'}</span>
        </div>
        <div className="font-mono text-[10px] text-muted-foreground">треклет {tr.tracklet_id}</div>
      </div>
    </div>
  )
}

function CaseCard({ runId, c }: { runId: number; c: ErrorCase }) {
  const merge = c.kind === 'false_merge'
  return (
    <Card className="gap-2 py-3">
      <CardContent className="space-y-2.5 px-3">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-sm">
          <Badge variant={merge ? 'destructive' : 'secondary'}>{merge ? 'склейка' : 'дробление'}</Badge>
          <span className="inline-flex flex-wrap items-center gap-1">
            {merge ? 'Глобальный ID' : 'Глобальные ID'} {c.global_ids.map((g) => <IdChip key={g} gid={g} />)}
          </span>
          <span className="text-muted-foreground">
            {merge ? 'содержит GT-людей' : 'у GT-человека'} <span className="font-mono text-foreground">{c.gt_person_ids.join(', ') || '—'}</span>
          </span>
          <span className="font-mono text-xs text-muted-foreground">
            {c.camera_id} · {formatClock(c.t)}
          </span>
          <span className="ml-auto">
            <PlayerLink runId={runId} t={c.t} gid={c.global_ids[0]} camera={c.camera_id} label="Открыть в плеере" />
          </span>
        </div>
        <div className="grid gap-1.5 sm:grid-cols-2 2xl:grid-cols-3">
          {c.tracklets.map((tr) => (
            <Tracklet key={tr.tracklet_id} runId={runId} tr={tr} />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function CaseList({ runId, cases, kind }: { runId: number; cases: ErrorCase[]; kind: 'merges' | 'splits' }) {
  const [shown, setShown] = useState(PAGE)
  if (!cases.length)
    return (
      <EmptyState
        title={kind === 'merges' ? 'Ложных склеек нет' : 'Ложных дроблений нет'}
        hint={kind === 'merges' ? 'Ни один глобальный ID не объединяет разных GT-людей.' : 'Каждый GT-человек получил один глобальный ID.'}
      />
    )
  return (
    <div className="space-y-2">
      {cases.slice(0, shown).map((c, i) => (
        <CaseCard key={`${c.global_ids.join('-')}-${c.gt_person_ids.join('-')}-${i}`} runId={runId} c={c} />
      ))}
      {shown < cases.length && (
        <Button variant="outline" size="sm" onClick={() => setShown((n) => n + PAGE)}>
          Показать ещё {Math.min(PAGE, cases.length - shown)} из {cases.length - shown}
        </Button>
      )}
    </div>
  )
}

function Body({ run }: { run: RunOut }) {
  const errors = useRunErrors(run.id)
  const search = useSearch({ from: '/runs/$runId/errors' })
  const navigate = useNavigate({ from: '/runs/$runId/errors' })
  const kind = search.kind ?? 'merges'
  const km = run.key_metrics
  return (
    <QueryView query={errors} loading={<LoadingBlock rows={8} />}>
      {(e) => (
        <div className="space-y-7">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <MetricTile metric="false_merges" value={km.false_merges ?? e.false_merges.length} />
            <MetricTile metric="false_splits" value={km.false_splits ?? e.false_splits.length} />
            <MetricTile metric="handover_success" value={km.handover_success} />
            <MetricTile metric="pairwise_f1" value={km.pairwise_f1} />
          </div>
          <Section
            title={<MetricLabel metric="handover_success" label="Переходы камера → камера" />}
            description="Переходы GT-людей между камерами: в скольких глобальный ID сохранился"
          >
            <Card className="py-3">
              <CardContent className="px-4">
                <TransitionMatrix transitions={e.transitions} />
              </CardContent>
            </Card>
          </Section>
          <Section
            title="Ошибки идентичности"
            description="Треклет сопоставлен GT-человеку по большинству рамок с IoU ≥ 0,5 (§9.3); список ограничен 100 случаями каждого вида"
            actions={
              <div className="flex gap-1" role="tablist">
                {(['merges', 'splits'] as const).map((k) => (
                  <Button
                    key={k}
                    size="sm"
                    role="tab"
                    aria-selected={kind === k}
                    variant={kind === k ? 'default' : 'outline'}
                    onClick={() => void navigate({ search: { kind: k }, replace: true })}
                  >
                    {k === 'merges' ? 'Ложные склейки' : 'Ложные дробления'}
                    <Badge variant="secondary" className="ml-1 h-4 px-1 font-mono text-[10px]">
                      {k === 'merges' ? e.false_merges.length : e.false_splits.length}
                    </Badge>
                  </Button>
                ))}
              </div>
            }
          >
            <CaseList key={kind} runId={run.id} kind={kind} cases={kind === 'merges' ? e.false_merges : e.false_splits} />
          </Section>
        </div>
      )}
    </QueryView>
  )
}

/** `/runs/:id/errors` — false merges / splits with crops and player links, handover transition heatmap. */
export function ErrorsPage() {
  const { runId } = useParams({ from: '/runs/$runId/errors' })
  const id = Number(runId)
  const run = useRun(id)
  if (!Number.isInteger(id)) return <EmptyState title="Некорректный номер прогона" />
  if (run.isPending) return <LoadingBlock rows={8} />
  if (run.isError) return <ErrorState error={run.error} onRetry={() => void run.refetch()} />
  const r = run.data
  return (
    <div className="space-y-6">
      <PageHeader
        title={`Разбор ошибок · ${r.name}`}
        description={
          <span>
            <span className="font-mono">#{r.id}</span> · {r.dataset}/<span className="font-mono">{r.sequence}</span> · {r.mode}
          </span>
        }
        actions={
          <>
            <Button asChild size="sm" variant="outline">
              <Link to="/runs/$runId" params={{ runId }}>
                <ArrowLeft /> К прогону
              </Link>
            </Button>
            <Button asChild size="sm" variant="outline" disabled={r.status !== 'done'}>
              <Link to="/runs/$runId/player" params={{ runId }}>
                <Film /> Плеер
              </Link>
            </Button>
          </>
        }
      />
      {r.status === 'done' ? <Body run={r} /> : <EmptyState title="Прогон ещё не завершён" hint="Разбор ошибок появится после завершения прогона." />}
    </div>
  )
}
