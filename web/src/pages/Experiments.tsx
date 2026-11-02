import { useMemo, useState } from 'react'
import { Link, useNavigate, useSearch } from '@tanstack/react-router'
import { ArrowDown, ArrowUp, GitCompare, Search, Trash2, X } from 'lucide-react'
import { toast } from 'sonner'
import { errorText } from '@/api/client'
import { useDatasets, useDeleteRun, useRuns } from '@/api/hooks'
import type { KeyMetrics, RunOut } from '@/api/types.gen'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Confirm } from '@/components/app/ConfirmButton'
import { DiffChips } from '@/components/app/DiffChips'
import { MetricTooltip } from '@/components/app/Metric'
import { PageHeader } from '@/components/app/PageHeader'
import { EmptyState, LoadingBlock, QueryView } from '@/components/app/States'
import { SplitBadge, StatusBadge } from '@/components/app/StatusBadge'
import { formatDate } from '@/lib/format'
import { bestIndex, formatMetric, metricDef } from '@/lib/metrics'
import { cn } from '@/lib/utils'

const COLUMNS: (keyof KeyMetrics)[] = [
  'cross_hota',
  'cross_idf1',
  'per_camera_idf1',
  'pairwise_f1',
  'false_merges',
  'false_splits',
  'handover_success',
  'reid_rank1',
  'fps',
  'peak_rss_gb',
]

const ALL = '__all__'

function FilterSelect({ label, value, options, onChange }: { label: string; value: string | undefined; options: [string, string][]; onChange: (v: string | undefined) => void }) {
  return (
    <Select value={value ?? ALL} onValueChange={(v) => onChange(v === ALL ? undefined : v)}>
      <SelectTrigger size="sm" className={cn('w-auto min-w-32', value && 'border-brand/50')} aria-label={label}>
        <span className="text-muted-foreground">{label}:</span>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL}>все</SelectItem>
        {options.map(([v, l]) => (
          <SelectItem key={v} value={v}>
            {l}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function sortRuns(runs: RunOut[], sort: string | undefined, desc: boolean | undefined): RunOut[] {
  if (!sort) return runs
  const dir = desc ? -1 : 1
  const val = (r: RunOut): number | string | null =>
    sort === 'id' ? r.id : sort === 'name' ? r.name : ((r.key_metrics[sort as keyof KeyMetrics] as number | null) ?? null)
  return [...runs].sort((a, b) => {
    const x = val(a)
    const y = val(b)
    if (x === null) return 1
    if (y === null) return -1
    return (x < y ? -1 : x > y ? 1 : 0) * dir
  })
}

export function ExperimentsPage() {
  const search = useSearch({ from: '/experiments' })
  const navigate = useNavigate({ from: '/experiments' })
  const setSearch = (patch: Partial<typeof search>) => void navigate({ search: (s) => ({ ...s, ...patch }), replace: true })
  const runs = useRuns({ dataset: search.dataset, split: search.split, mode: search.mode, status: search.status, sweep_id: search.sweep })
  const datasets = useDatasets()
  const del = useDeleteRun()
  const [selected, setSelected] = useState<number[]>([])

  const rows = useMemo(() => {
    const q = search.q?.toLowerCase().trim()
    const filtered = (runs.data ?? []).filter(
      (r) => !q || `${r.id} ${r.name} ${r.dataset} ${r.sequence} ${Object.keys(r.config_diff).join(' ')}`.toLowerCase().includes(q),
    )
    return sortRuns(filtered, search.sort, search.desc)
  }, [runs.data, search.q, search.sort, search.desc])

  // "Best" is only meaningful within one dataset.
  const best = useMemo<Record<string, number | undefined>>(
    () =>
      !search.dataset
        ? {}
        : Object.fromEntries(COLUMNS.map((c) => [c, rows[bestIndex(c, rows.map((r) => (r.status === 'done' ? r.key_metrics[c] : null)))]?.id])),
    [rows, search.dataset],
  )
  const sweeps = [...new Set((runs.data ?? []).map((r) => r.sweep_id).filter((s): s is number => s != null))]
  const toggleSort = (key: string) => setSearch(search.sort === key ? { desc: !search.desc } : { sort: key, desc: metricDef(key)?.higherIsBetter !== false })
  const toggle = (id: number) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))
  const SortIcon = search.desc ? ArrowDown : ArrowUp

  const sortHead = (k: string, label: string) => (
    <th className="px-2 py-2 font-medium whitespace-nowrap">
      <button className="inline-flex items-center gap-0.5 hover:text-foreground" onClick={() => toggleSort(k)}>
        {label}
        {search.sort === k && <SortIcon className="size-3" />}
      </button>
    </th>
  )

  return (
    <>
      <PageHeader
        title="Эксперименты"
        description="Все прогоны с ключевыми метриками (× — межкамерные); при фильтре по датасету лучшее значение выделено. Выберите 2+ прогона для сравнения"
        actions={
          <>
            {selected.length > 0 && (
              <Button variant="ghost" size="sm" onClick={() => setSelected([])}>
                <X /> Снять выбор
              </Button>
            )}
            <Button size="sm" disabled={selected.length < 2} onClick={() => void navigate({ to: '/compare', search: { runs: selected.join(',') } })}>
              <GitCompare /> Сравнить{selected.length ? ` (${selected.length})` : ''}
            </Button>
          </>
        }
      />
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="pointer-events-none absolute top-2 left-2 size-4 text-muted-foreground" />
          <Input className="h-8 w-56 pl-8" placeholder="Поиск: имя, датасет, ключ" value={search.q ?? ''} onChange={(e) => setSearch({ q: e.target.value || undefined })} />
        </div>
        <FilterSelect label="Датасет" value={search.dataset} options={(datasets.data ?? []).map((d) => [d.id, d.title])} onChange={(dataset) => setSearch({ dataset })} />
        <FilterSelect label="Сплит" value={search.split} options={['tune', 'test', 'all'].map((s) => [s, s])} onChange={(split) => setSearch({ split })} />
        <FilterSelect label="Режим" value={search.mode} options={[['full', 'full'], ['oracle', 'oracle']]} onChange={(mode) => setSearch({ mode })} />
        <FilterSelect
          label="Статус"
          value={search.status}
          options={[['done', 'готово'], ['running', 'выполняется'], ['queued', 'в очереди'], ['failed', 'ошибка'], ['cancelled', 'отменено']]}
          onChange={(status) => setSearch({ status })}
        />
        <FilterSelect
          label="Sweep"
          value={search.sweep == null ? undefined : String(search.sweep)}
          options={(search.sweep != null && !sweeps.includes(search.sweep) ? [...sweeps, search.sweep] : sweeps).map((s) => [String(s), `#${s}`])}
          onChange={(v) => setSearch({ sweep: v ? Number(v) : undefined })}
        />
        <span className="ml-auto text-xs text-muted-foreground tabular-nums">{rows.length} прогонов</span>
      </div>
      <QueryView
        query={runs}
        loading={<LoadingBlock rows={10} />}
        isEmpty={(rs) => rs.length === 0}
        empty={
          <EmptyState
            title="Прогонов не найдено"
            hint="Измените фильтры или запустите новый прогон."
            action={
              <Button asChild size="sm" variant="outline">
                <Link to="/runs/new">Новый прогон</Link>
              </Button>
            }
          />
        }
      >
        {() => (
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-sm">
              <thead className="border-b bg-muted/40 text-left text-xs text-muted-foreground">
                <tr>
                  <th className="w-8 px-2 py-2" />
                  {sortHead('id', '#')}
                  {sortHead('name', 'Прогон')}
                                    {COLUMNS.map((c) => (
                    <th key={c} className="px-2 py-2 text-right font-medium whitespace-nowrap">
                      <MetricTooltip metric={c}>
                        <button className="inline-flex cursor-help items-center gap-0.5 hover:text-foreground" onClick={() => toggleSort(c)}>
                          {search.sort === c && <SortIcon className="size-3" />}
                          {metricDef(c)?.label ?? c}
                        </button>
                      </MetricTooltip>
                    </th>
                  ))}
                  <th className="px-2 py-2 font-medium">Конфиг-диф</th>
                  <th className="w-8" />
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} className={cn('border-b last:border-0 hover:bg-muted/40', selected.includes(r.id) && 'bg-brand/5')}>
                    <td className="px-2 py-1.5">
                      <Checkbox checked={selected.includes(r.id)} onCheckedChange={() => toggle(r.id)} aria-label={`Выбрать прогон ${r.id}`} />
                    </td>
                    <td className="px-2 py-1.5 font-mono text-xs text-muted-foreground">{r.id}</td>
                    <td className="max-w-64 px-2 py-1.5">
                      <Link
                        to="/runs/$runId"
                        params={{ runId: String(r.id) }}
                        className="block truncate font-medium hover:underline"
                        title={`${r.name} · создан ${formatDate(r.created_at)}`}
                      >
                        {r.name}
                      </Link>
                      <div className="flex items-center gap-1 overflow-hidden text-xs whitespace-nowrap text-muted-foreground">
                        {r.status !== 'done' && <StatusBadge status={r.status} className="h-4 px-1.5" />}
                        <span className="truncate">
                          {r.dataset}/<span className="font-mono">{r.sequence}</span>
                        </span>
                        <SplitBadge split={r.split} className="h-4 px-1.5" />
                        <Badge variant="outline" className="h-4 px-1.5 font-mono font-normal">
                          {r.mode}
                        </Badge>
                        {r.sweep_id != null && (
                          <button className="hover:text-foreground hover:underline" onClick={() => setSearch({ sweep: r.sweep_id ?? undefined })}>
                            sweep #{r.sweep_id}
                          </button>
                        )}
                      </div>
                    </td>
                    {COLUMNS.map((c) => (
                      <td key={c} className={cn('px-2 py-1.5 text-right font-mono text-xs whitespace-nowrap tabular-nums', best[c] === r.id && 'font-semibold text-good')}>
                        {formatMetric(c, r.key_metrics[c])}
                      </td>
                    ))}
                    <td className="px-2 py-1.5">
                      <DiffChips diff={r.config_diff} />
                    </td>
                    <td className="px-1">
                      <Confirm
                        trigger={
                          <Button variant="ghost" size="icon-xs" aria-label={`Удалить прогон ${r.id}`} disabled={r.status === 'running' || r.status === 'queued'}>
                            <Trash2 />
                          </Button>
                        }
                        title={`Удалить прогон #${r.id}?`}
                        description="Будут удалены результаты прогона (треки, метрики, логи). Кеш тяжёлых стадий сохранится."
                        confirmLabel="Удалить"
                        onConfirm={() =>
                          del.mutate(r.id, {
                            onSuccess: () => {
                              setSelected((s) => s.filter((x) => x !== r.id))
                              toast.success(`Прогон #${r.id} удалён`)
                            },
                            onError: (e) => toast.error(errorText(e)),
                          })
                        }
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </QueryView>
    </>
  )
}
