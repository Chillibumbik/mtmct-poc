import type { TransitionCell } from '@/api/types.gen'
import { cn } from '@/lib/utils'

/**
 * Camera → camera handover heatmap: share of GT transitions where the global id was kept (kept / total).
 * Single-hue sequential scale (colour-blind safe); text carries the numbers.
 */
export function TransitionMatrix({ transitions }: { transitions: TransitionCell[] }) {
  const cams = [...new Set(transitions.flatMap((t) => [t.from_camera, t.to_camera]))].sort()
  if (!cams.length) return <div className="text-xs text-muted-foreground">В GT нет переходов людей между камерами.</div>
  const cell = new Map(transitions.map((t) => [`${t.from_camera}→${t.to_camera}`, t]))
  const kept = transitions.reduce((a, t) => a + t.kept, 0)
  const total = transitions.reduce((a, t) => a + t.total, 0)
  return (
    <div className="space-y-2">
      <div className="overflow-x-auto">
        <table className="text-xs tabular-nums">
          <thead>
            <tr>
              <th className="px-2 py-1 text-left font-normal text-muted-foreground">из ↓ в →</th>
              {cams.map((c) => (
                <th key={c} className="px-0.5 py-1 font-mono font-medium">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cams.map((a) => (
              <tr key={a}>
                <th className="px-2 py-0.5 text-left font-mono font-medium">{a}</th>
                {cams.map((b) => {
                  const t = cell.get(`${a}→${b}`)
                  const rate = t && t.total ? t.kept / t.total : null
                  return (
                    <td key={b} className="p-0.5">
                      <div
                        className={cn('flex h-11 w-16 flex-col items-center justify-center rounded', rate === null && 'bg-muted/40 text-muted-foreground')}
                        style={
                          rate === null
                            ? undefined
                            : { background: `color-mix(in oklab, var(--chart-1) ${Math.round(12 + rate * 78)}%, transparent)`, color: rate > 0.7 ? '#fff' : undefined }
                        }
                        title={t ? `${a} → ${b}: ID сохранён в ${t.kept} из ${t.total} переходов` : `${a} → ${b}: переходов нет`}
                      >
                        {t ? (
                          <>
                            <span className="font-semibold">{rate === null ? '—' : `${Math.round(rate * 100)}%`}</span>
                            <span className="text-[10px] opacity-85">
                              {t.kept}/{t.total}
                            </span>
                          </>
                        ) : (
                          '·'
                        )}
                      </div>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          0%
          <span className="inline-block h-2 w-24 rounded-full" style={{ background: 'linear-gradient(90deg, color-mix(in oklab, var(--chart-1) 12%, transparent), var(--chart-1))' }} />
          100% ID сохранён
        </span>
        <span>
          Всего: {kept} из {total} ({total ? Math.round((kept / total) * 100) : 0}%)
        </span>
      </div>
    </div>
  )
}
