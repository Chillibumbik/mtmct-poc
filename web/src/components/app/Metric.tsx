import type { ReactNode } from 'react'
import { Info } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { directionHint, formatMetric, metricDef } from '@/lib/metrics'
import { cn } from '@/lib/utils'

export function MetricTooltip({ metric, children }: { metric: string; children: ReactNode }) {
  const def = metricDef(metric)
  if (!def) return <>{children}</>
  const hint = directionHint(def)
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent className="max-w-72 text-xs leading-relaxed">
        <div className="mb-1 flex items-center justify-between gap-3 font-semibold">
          <span>{def.label}</span>
          {hint && <span className="font-normal opacity-80">{hint}</span>}
        </div>
        {def.description}
      </TooltipContent>
    </Tooltip>
  )
}

/** Metric name with an info tooltip (definition + direction). */
export function MetricLabel({ metric, label, className }: { metric: string; label?: string; className?: string }) {
  const def = metricDef(metric)
  return (
    <MetricTooltip metric={metric}>
      <span className={cn('inline-flex cursor-help items-center gap-1 decoration-dotted underline-offset-4', className)} tabIndex={0}>
        {label ?? def?.label ?? metric}
        {def && <Info className="size-3 opacity-40" aria-hidden />}
      </span>
    </MetricTooltip>
  )
}

export function MetricTile({
  metric,
  value,
  label,
  sub,
  className,
}: {
  metric: string
  value: number | null | undefined
  label?: string
  sub?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('rounded-lg border bg-card px-3 py-2.5', className)}>
      <div className="text-xs text-muted-foreground">
        <MetricLabel metric={metric} label={label} />
      </div>
      <div className="mt-1 text-xl font-semibold tabular-nums tracking-tight">{formatMetric(metric, value)}</div>
      {sub && <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>}
    </div>
  )
}

/** Compact grid of metric tiles from a metrics.json section. */
export function MetricGrid({ section, keys, className }: { section: Record<string, unknown> | undefined; keys: readonly string[]; className?: string }) {
  return (
    <div className={cn('grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6', className)}>
      {keys.map((k) => (
        <MetricTile key={k} metric={k} value={typeof section?.[k] === 'number' ? (section[k] as number) : null} />
      ))}
    </div>
  )
}
