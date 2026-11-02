import { useState } from 'react'
import { Link } from '@tanstack/react-router'
import { Download, ExternalLink, Film, KeyRound, Lock, PlayCircle, RotateCw } from 'lucide-react'
import { toast } from 'sonner'
import { errorText } from '@/api/client'
import { useDatasets, useFetchDataset, useJobEvents } from '@/api/hooks'
import type { DatasetOut, SequenceOut } from '@/api/types.gen'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { PageHeader } from '@/components/app/PageHeader'
import { EmptyState, LoadingBlock, QueryView } from '@/components/app/States'
import { OverlapBadge, SplitBadge, StatusBadge } from '@/components/app/StatusBadge'
import { STAGE_LABELS, formatEta } from '@/lib/format'
import { formatSeconds } from '@/lib/metrics'
import { UploadDialog } from './UploadDialog'

function JobProgress({ jobId }: { jobId: number }) {
  const p = useJobEvents(jobId)
  const last = p.logs.at(-1)
  return (
    <div className="space-y-1.5 rounded-md border bg-muted/30 p-2.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">{p.stage ? (STAGE_LABELS[p.stage] ?? p.stage) : 'Ожидание воркера…'}</span>
        <span className="tabular-nums text-muted-foreground">
          {Math.round(p.progress * 100)}% · ETA {formatEta(p.etaS)}
        </span>
      </div>
      <Progress value={p.progress * 100} className="h-1.5" />
      {last && <div className="truncate font-mono text-[11px] text-muted-foreground">{last.message}</div>}
      <Link to="/jobs" search={{ job: jobId }} className="text-xs text-brand hover:underline">
        Задача #{jobId}
      </Link>
    </div>
  )
}

function SequenceRow({ ds, seq }: { ds: DatasetOut; seq: SequenceOut }) {
  return (
    <div className="space-y-2 rounded-md border p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-medium">{seq.name}</span>
        <span className="text-xs text-muted-foreground tabular-nums">
          {formatSeconds(seq.duration_s)} · {seq.cameras.length} кам.
        </span>
        {Object.keys(seq.splits).map((s) => (
          <SplitBadge key={s} split={s} />
        ))}
        <Badge variant="outline" className={seq.has_gt ? 'font-normal' : 'font-normal text-muted-foreground line-through'}>
          GT
        </Badge>
        <Badge variant="outline" className={seq.has_calibration ? 'font-normal' : 'font-normal text-muted-foreground line-through'}>
          калибровка
        </Badge>
        {!seq.media_ready && (
          <Badge variant="outline" className="font-normal text-warn">
            нет прокси-видео
          </Badge>
        )}
        <Button asChild size="xs" variant="ghost" className="ml-auto">
          <Link to="/runs/new" search={{ dataset: ds.id, sequence: seq.name }}>
            <PlayCircle /> Прогон
          </Link>
        </Button>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(7rem,1fr))] gap-1.5">
        {seq.cameras.map((c) => (
          <figure key={c} className="relative overflow-hidden rounded border bg-muted">
            {seq.preview_urls[c] ? (
              <img src={seq.preview_urls[c]} alt={`${seq.name} ${c}`} loading="lazy" className="aspect-video w-full object-cover" />
            ) : (
              <div className="flex aspect-video items-center justify-center text-muted-foreground">
                <Film className="size-4" />
              </div>
            )}
            <figcaption className="absolute bottom-0.5 left-1 rounded bg-black/60 px-1 font-mono text-[10px] text-white">{c}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  )
}

function DatasetCard({ ds }: { ds: DatasetOut }) {
  const fetch = useFetchDataset()
  const busy = ds.status === 'downloading' || ds.status === 'preparing' || ds.job_id != null
  const canFetch = !busy && ds.status !== 'ready' && !ds.custom
  const onFetch = () =>
    fetch.mutate(ds.id, {
      onSuccess: (job) => toast.success(`Задача #${job.id} поставлена в очередь`),
      onError: (e) => toast.error(errorText(e)),
    })
  return (
    <Card className="gap-3 py-4">
      <CardHeader className="px-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
              {ds.title}
              <OverlapBadge overlap={ds.overlap} />
              {!ds.real && (
                <Badge variant="secondary" className="font-normal">
                  синтетика
                </Badge>
              )}
              {ds.custom && (
                <Badge variant="secondary" className="font-normal">
                  свой набор
                </Badge>
              )}
            </CardTitle>
            <div className="mt-0.5 font-mono text-xs text-muted-foreground">{ds.id}</div>
          </div>
          <StatusBadge status={ds.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-3 px-4">
        <p className="text-sm text-muted-foreground">{ds.description}</p>
        <dl className="grid grid-cols-3 gap-2 text-xs">
          <div>
            <dt className="text-muted-foreground">Камеры</dt>
            <dd className="tabular-nums">{ds.n_cameras}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Объём</dt>
            <dd className="tabular-nums">{ds.download_gb ? `${ds.download_gb} ГБ` : '—'}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Лицензия</dt>
            <dd className="truncate" title={ds.licence}>
              {ds.licence}
            </dd>
          </div>
        </dl>
        {ds.requires_auth && ds.status !== 'ready' && (
          <div className="flex gap-2 rounded-md border border-warn/40 bg-warn/5 p-2 text-xs">
            <KeyRound className="size-3.5 shrink-0 text-warn" />
            Требуется вход на Hugging Face и принятие условий датасета владельцем; токен задаётся на стороне сервера.
          </div>
        )}
        {ds.status === 'error' && ds.status_message && <div className="text-xs text-bad">{ds.status_message}</div>}
        {ds.custom && ds.sequences.length === 0 && ds.job_id == null && (
          <div className="text-xs text-muted-foreground">Добавьте видео кнопкой «Загрузить свои видео».</div>
        )}
        {ds.job_id != null && <JobProgress jobId={ds.job_id} />}
        <div className="flex flex-wrap items-center gap-2">
          {canFetch && (
            <Button size="sm" onClick={onFetch} disabled={fetch.isPending}>
              {ds.status === 'error' ? <RotateCw /> : ds.requires_auth ? <Lock /> : <Download />}
              {ds.status === 'error' ? 'Повторить' : ds.status === 'downloaded' ? 'Подготовить' : 'Скачать и подготовить'}
            </Button>
          )}
          {ds.source_url && (
            <Button asChild size="sm" variant="ghost">
              <a href={ds.source_url} target="_blank" rel="noreferrer">
                Источник <ExternalLink />
              </a>
            </Button>
          )}
        </div>
        {ds.sequences.length > 0 && (
          <div className="space-y-2">
            {ds.sequences.map((s) => (
              <SequenceRow key={s.name} ds={ds} seq={s} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export function DataPage() {
  const datasets = useDatasets()
  const [uploadOpen, setUploadOpen] = useState(false)
  return (
    <>
      <PageHeader
        title="Данные"
        description="Открытые датасеты и собственные наборы видео; скачивание и подготовка выполняются задачами"
        actions={<UploadDialog open={uploadOpen} onOpenChange={setUploadOpen} />}
      />
      <QueryView
        query={datasets}
        loading={<LoadingBlock rows={6} />}
        isEmpty={(d) => d.length === 0}
        empty={<EmptyState title="Каталог датасетов пуст" hint="Проверьте configs/datasets на сервере." />}
      >
        {(ds) => (
          <div className="gap-3 xl:columns-2 [&>*]:mb-3 [&>*]:break-inside-avoid">
            {ds.map((d) => (
              <DatasetCard key={d.id} ds={d} />
            ))}
          </div>
        )}
      </QueryView>
    </>
  )
}
