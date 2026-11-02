import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { formatValue } from '@/lib/format'

/** Config overrides (dotted key → value) as compact chips; the rest in a tooltip. */
export function DiffChips({ diff, max = 3 }: { diff: Record<string, unknown>; max?: number }) {
  const entries = Object.entries(diff)
  if (entries.length === 0) return <span className="text-xs text-muted-foreground">базовый</span>
  const chip = ([k, v]: [string, unknown]) => (
    <Badge key={k} variant="outline" className="max-w-56 font-mono text-[10.5px] font-normal" title={`${k} = ${formatValue(v)}`}>
      <span className="truncate">
        <span className="text-muted-foreground">{k.split('.').slice(-2).join('.')}=</span>
        {formatValue(v)}
      </span>
    </Badge>
  )
  return (
    <div className="flex flex-wrap gap-1">
      {entries.slice(0, max).map(chip)}
      {entries.length > max && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Badge variant="secondary" className="cursor-help font-normal">
              +{entries.length - max}
            </Badge>
          </TooltipTrigger>
          <TooltipContent className="max-w-96 font-mono text-[11px]">
            {entries.slice(max).map(([k, v]) => (
              <div key={k}>
                {k} = {formatValue(v)}
              </div>
            ))}
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  )
}
