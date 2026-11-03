import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearch } from '@tanstack/react-router'
import { ArrowLeft, Bug, Info } from 'lucide-react'
import { urls } from '@/api/client'
import { useDatasets, usePerson, useRun } from '@/api/hooks'
import type { RunOut, SequenceOut } from '@/api/types.gen'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { PageHeader } from '@/components/app/PageHeader'
import { EmptyState, ErrorState, LoadingBlock } from '@/components/app/States'
import { BevView } from '@/features/bev/BevView'
import { CameraTile } from './CameraTile'
import { MasterClock, shiftSpeed, stepFrame } from './clock'
import { colorForId } from './colors'
import { Controls } from './Controls'
import { formatClock } from './format'
import { DIFF_COLORS, type OverlayMode } from './overlay'
import { PersonPanel } from './PersonPanel'
import { useTrackIndex } from './useTracks'
import { runTimeRange, windowsAt } from './windows'

const ROUTE = '/runs/$runId/player'

function gridShape(n: number): { cols: number; rows: number } {
  const cols = n <= 1 ? 1 : n <= 4 ? 2 : n <= 9 ? 3 : 4
  return { cols, rows: Math.ceil(n / cols) }
}

function DiffLegend() {
  const item = (color: string, line: string, label: string) => (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block h-3 w-4 rounded-[2px] border-2 bg-black/70" style={{ borderColor: color, borderStyle: line }} />
      {label}
    </span>
  )
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {item(DIFF_COLORS.error, 'solid', 'ID расходится с основным ID этого GT-человека')}
      {item(DIFF_COLORS.ok, 'solid', 'совпадает')}
      {item(DIFF_COLORS.none, 'dashed', 'рамка без GT')}
      {item(DIFF_COLORS.none, 'dotted', 'GT-человек пропущен')}
    </div>
  )
}

function PlayerView({ run, seq }: { run: RunOut; seq: SequenceOut }) {
  const search = useSearch({ from: ROUTE })
  const navigate = useNavigate({ from: ROUTE })
  const range = useMemo(() => runTimeRange(run.config, seq, run.split), [run.config, run.split, seq])
  const [clock] = useState(() => new MasterClock(range[0], range[1], search.t ?? range[0]))
  const [windows, setWindows] = useState(() => windowsAt(clock.time(), 1, range))
  const cameras = seq.cameras
  const { indices, majority, error } = useTrackIndex(run.id, cameras, windows)
  const mode: OverlayMode = seq.has_gt ? (search.mode ?? 'pred') : 'pred'
  const tab = search.tab ?? 'grid'
  const selected = search.gid ?? null
  const person = usePerson(run.id, selected ?? -1, selected !== null)
  const frameS = 1 / Math.max(1, ...[...indices.values()].map((i) => i.meta.fps))

  const select = (gid: number | null) => void navigate({ search: (p) => ({ ...p, gid: gid ?? undefined }), replace: true })

  // master clock drives everything; UI subscribers are notified at ~10 Hz
  useEffect(() => {
    let raf = 0
    const loop = () => {
      clock.tick()
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf)
      clock.pause()
    }
  }, [clock])

  // fetch box windows ahead of the playhead (each once; kept for the page's life)
  useEffect(
    () =>
      clock.subscribe(() => {
        const { t, speed } = clock.getSnapshot()
        setWindows((prev) => {
          const add = windowsAt(t, speed, range).filter((k) => !prev.includes(k))
          return add.length ? [...prev, ...add] : prev
        })
      }),
    [clock, range],
  )

  // paused position → `?t=` (deep link); not while playing
  useEffect(() => {
    let timer = 0
    const unsub = clock.subscribe(() => {
      window.clearTimeout(timer)
      const s = clock.getSnapshot()
      if (s.playing) return
      timer = window.setTimeout(() => void navigate({ search: (p) => ({ ...p, t: Math.round(s.t * 100) / 100 }), replace: true }), 400)
    })
    return () => {
      unsub()
      window.clearTimeout(timer)
    }
  }, [clock, navigate])

  // `?t=` changed from outside (back/forward, link from the error browser) → seek
  useEffect(() => {
    if (search.t != null && !clock.playing && Math.abs(search.t - clock.time()) > 0.02) clock.seek(search.t)
  }, [search.t, clock])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey) return
      if ((e.target as HTMLElement).closest('input, textarea, select, [contenteditable="true"], [role="slider"]')) return
      if (e.key === ' ') clock.toggle()
      else if (e.key === 'ArrowLeft') stepFrame(clock, e.shiftKey ? -1 : -frameS)
      else if (e.key === 'ArrowRight') stepFrame(clock, e.shiftKey ? 1 : frameS)
      else if (e.key === '[') clock.setSpeed(shiftSpeed(clock.speed, -1))
      else if (e.key === ']') clock.setSpeed(shiftSpeed(clock.speed, 1))
      else if (e.key === 'Escape') void navigate({ search: (p) => ({ ...p, gid: undefined }), replace: true })
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [clock, frameS, navigate])

  const { cols, rows } = gridShape(cameras.length)
  const first = indices.get(cameras[0])?.meta
  const aspect = first ? first.width / first.height : 4 / 3
  const marks = (person.data?.segments ?? []).map((s) => ({ from: s.t_start, to: s.t_end, color: colorForId(selected ?? 0) }))

  return (
    <div className="flex min-h-[calc(100dvh-2.5rem)] flex-col">
      <PageHeader
        title={`Плеер · ${run.name}`}
        description={
          <span className="flex flex-wrap items-center gap-1.5">
            <span className="font-mono">#{run.id}</span>
            <span>
              {run.dataset}/<span className="font-mono">{run.sequence}</span>
            </span>
            <Badge variant="outline" className="font-mono font-normal">
              {run.mode}
            </Badge>
            <span>
              · {formatClock(range[0])}–{formatClock(range[1])} · {cameras.length} кам.
            </span>
          </span>
        }
        actions={
          <>
            <Button asChild size="sm" variant="outline">
              <Link to="/runs/$runId" params={{ runId: String(run.id) }}>
                <ArrowLeft /> К прогону
              </Link>
            </Button>
            <Button asChild size="sm" variant="outline">
              <Link to="/runs/$runId/errors" params={{ runId: String(run.id) }}>
                <Bug /> Ошибки
              </Link>
            </Button>
          </>
        }
      />
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <ToggleGroup
          type="single"
          size="sm"
          variant="outline"
          value={tab}
          onValueChange={(v) => v && void navigate({ search: (p) => ({ ...p, tab: v as 'grid' | 'bev' }), replace: true })}
          aria-label="Вид"
        >
          <ToggleGroupItem value="grid" className="px-3">
            Камеры
          </ToggleGroupItem>
          <ToggleGroupItem value="bev" className="px-3">
            Вид сверху
          </ToggleGroupItem>
        </ToggleGroup>
        <ToggleGroup
          type="single"
          size="sm"
          variant="outline"
          value={mode}
          onValueChange={(v) => v && void navigate({ search: (p) => ({ ...p, mode: v as OverlayMode }), replace: true })}
          aria-label="Режим наложения"
        >
          <ToggleGroupItem value="pred" className="px-3">
            Предсказания
          </ToggleGroupItem>
          <ToggleGroupItem value="gt" className="px-3" disabled={!seq.has_gt}>
            GT
          </ToggleGroupItem>
          <ToggleGroupItem value="diff" className="px-3" disabled={!seq.has_gt}>
            Расхождения
          </ToggleGroupItem>
        </ToggleGroup>
        {mode === 'diff' && <DiffLegend />}
        {mode === 'gt' && <span className="text-xs text-muted-foreground">Цвет GT-человека — цвет глобального ID, который он чаще всего получает.</span>}
      </div>
      {!seq.media_ready && (
        <div className="mb-3 flex items-start gap-2 rounded-lg border px-3 py-2 text-xs text-muted-foreground">
          <Info className="mt-0.5 size-3.5 shrink-0" />
          Прокси-видео для этой последовательности ещё не сделаны — рамки показываются без видео. Их создаёт подготовка датасета на странице «Данные».
        </div>
      )}
      {error != null && <ErrorState error={error} className="mb-3" />}
      <div className="flex flex-1 flex-col gap-4 lg:flex-row lg:items-start">
        <div className="min-w-0 flex-1">
          <div className={tab === 'grid' ? '' : 'hidden'}>
            <div
              className="mx-auto grid gap-2"
              style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, maxWidth: `max(480px, calc((100dvh - 290px) * ${(aspect * cols) / rows}))` }}
            >
              {cameras.map((c) => (
                <CameraTile
                  key={c}
                  camera={c}
                  src={seq.media_ready ? urls.media(run.dataset, run.sequence, c) : null}
                  poster={seq.preview_urls[c] ?? null}
                  index={indices.get(c)}
                  clock={clock}
                  mode={mode}
                  majority={majority}
                  selected={selected}
                  focused={search.camera === c}
                  onSelect={select}
                />
              ))}
            </div>
          </div>
          {tab === 'bev' && <BevView runId={run.id} clock={clock} selected={selected} onSelect={select} />}
        </div>
        {selected !== null && <PersonPanel runId={run.id} gid={selected} cameras={cameras} range={range} clock={clock} onClose={() => select(null)} />}
      </div>
      <Controls clock={clock} frameS={frameS} marks={marks} />
    </div>
  )
}

/** `/runs/:id/player?t=&gid=&camera=&mode=&tab=` — synchronised camera grid with overlays, BEV and person routes. */
export function PlayerPage() {
  const { runId } = useParams({ from: ROUTE })
  const id = Number(runId)
  const run = useRun(id)
  const datasets = useDatasets()
  if (!Number.isInteger(id)) return <EmptyState title="Некорректный номер прогона" />
  if (run.isPending || datasets.isPending) return <LoadingBlock rows={8} />
  if (run.isError) return <ErrorState error={run.error} onRetry={() => void run.refetch()} />
  if (datasets.isError) return <ErrorState error={datasets.error} onRetry={() => void datasets.refetch()} />
  const r = run.data
  const back = (
    <Button asChild size="sm" variant="outline">
      <Link to="/runs/$runId" params={{ runId }}>
        К прогону
      </Link>
    </Button>
  )
  if (r.status !== 'done') return <EmptyState title="Прогон ещё не завершён" hint="Плеер доступен после успешного завершения прогона." action={back} />
  const seq = datasets.data.find((d) => d.id === r.dataset)?.sequences.find((s) => s.name === r.sequence)
  if (!seq) return <EmptyState title="Последовательность не подготовлена" hint={`${r.dataset}/${r.sequence} нет среди готовых данных.`} action={back} />
  return <PlayerView key={r.id} run={r} seq={seq} />
}
