import { useSyncExternalStore } from 'react'
import { AlertTriangle, CornerDownRight, X } from 'lucide-react'
import { usePerson } from '@/api/hooks'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { QueryView } from '@/components/app/States'
import type { MasterClock } from './clock'
import { colorForId, textColorOn } from './colors'
import { formatClock } from './format'

export function IdChip({ gid, className }: { gid: number; className?: string }) {
  const c = colorForId(gid)
  return (
    <span className={`inline-flex items-center rounded px-1.5 font-mono text-xs font-semibold ${className ?? ''}`} style={{ background: c, color: textColorOn(c) }}>
      {gid}
    </span>
  )
}

/** Route of one global id (`/runs/{id}/person/{gid}`): per-camera timeline on the common clock, crops, jump-to. */
export function PersonPanel({
  runId,
  gid,
  cameras,
  range,
  clock,
  onClose,
}: {
  runId: number
  gid: number
  cameras: string[]
  range: [number, number]
  clock: MasterClock
  onClose: () => void
}) {
  const q = usePerson(runId, gid)
  const { t } = useSyncExternalStore(clock.subscribe, clock.getSnapshot)
  const span = Math.max(1e-6, range[1] - range[0])
  const pct = (x: number) => `${(Math.min(Math.max(x - range[0], 0), span) / span) * 100}%`
  const jump = (to: number) => {
    clock.pause()
    clock.seek(to)
  }
  return (
    <aside className="flex w-full flex-col gap-3 rounded-lg border bg-card p-3 lg:w-80 lg:shrink-0" aria-label={`Маршрут глобального ID ${gid}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-medium">
          Глобальный ID <IdChip gid={gid} />
        </div>
        <Button size="icon-xs" variant="ghost" onClick={onClose} aria-label="Закрыть маршрут">
          <X />
        </Button>
      </div>
      <QueryView query={q}>
        {(p) => {
          const segCams = cameras.filter((c) => p.segments.some((s) => s.camera_id === c))
          const total = p.segments.reduce((a, s) => a + (s.t_end - s.t_start), 0)
          return (
            <>
              <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                <span>GT:</span>
                {p.gt_person_ids.length ? p.gt_person_ids.map((g) => <Badge key={g} variant="outline" className="font-mono">{g}</Badge>) : <span>нет совпадений</span>}
                {p.gt_person_ids.length > 1 && (
                  <span className="inline-flex items-center gap-1 text-bad">
                    <AlertTriangle className="size-3" /> склейка разных людей
                  </span>
                )}
              </div>
              <div className="text-xs text-muted-foreground">
                {p.segments.length} {p.segments.length === 1 ? 'отрезок' : 'отрезков'} в {segCams.length} кам., всего {total.toFixed(1)} с
              </div>
              {/* timeline: one row per camera on the run's time range */}
              <div className="space-y-1">
                {segCams.map((c) => (
                  <div key={c} className="flex items-center gap-2">
                    <span className="w-7 shrink-0 font-mono text-[11px] text-muted-foreground">{c}</span>
                    <div className="relative h-4 flex-1 rounded-sm bg-muted">
                      {p.segments
                        .filter((s) => s.camera_id === c)
                        .map((s) => (
                          <button
                            key={`${s.t_start}-${s.tracklet_ids.join(',')}`}
                            type="button"
                            className="absolute inset-y-0 min-w-[3px] rounded-sm hover:brightness-110"
                            style={{ left: pct(s.t_start), width: `calc(${pct(s.t_end)} - ${pct(s.t_start)})`, background: colorForId(gid) }}
                            title={`${c}: ${formatClock(s.t_start)}–${formatClock(s.t_end)} — перейти`}
                            onClick={() => jump(s.t_start)}
                          />
                        ))}
                      <div className="pointer-events-none absolute inset-y-[-2px] w-px bg-foreground" style={{ left: pct(t) }} />
                    </div>
                  </div>
                ))}
              </div>
              <ol className="-mx-1 max-h-[calc(100dvh-420px)] min-h-24 space-y-1 overflow-y-auto">
                {p.segments.map((s) => (
                  <li key={`${s.camera_id}-${s.t_start}-${s.tracklet_ids.join(',')}`}>
                    <button
                      type="button"
                      onClick={() => jump(s.t_start)}
                      className="flex w-full items-center gap-2 rounded-md p-1 text-left hover:bg-muted"
                      title="Перейти к началу отрезка"
                    >
                      {s.crop_url ? (
                        <img src={s.crop_url} alt="" loading="lazy" className="h-14 w-auto max-w-12 shrink-0 rounded bg-muted object-cover" />
                      ) : (
                        <div className="h-14 w-8 shrink-0 rounded bg-muted" />
                      )}
                      <div className="min-w-0 flex-1 text-xs">
                        <div className="font-mono font-medium">{s.camera_id}</div>
                        <div className="tabular-nums text-muted-foreground">
                          {formatClock(s.t_start)}–{formatClock(s.t_end)}
                        </div>
                        <div className="truncate font-mono text-[10px] text-muted-foreground">треклет {s.tracklet_ids.join(', ')}</div>
                      </div>
                      <CornerDownRight className="size-3.5 shrink-0 text-muted-foreground" />
                    </button>
                  </li>
                ))}
              </ol>
            </>
          )
        }}
      </QueryView>
    </aside>
  )
}
