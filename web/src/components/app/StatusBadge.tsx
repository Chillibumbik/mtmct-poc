import { CheckCircle2, CircleDashed, CircleSlash, Download, Loader2, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { DatasetOut, JobOut } from '@/api/types.gen'
import { cn } from '@/lib/utils'

type Status = JobOut['status'] | DatasetOut['status']

const STATUS: Record<Status, { label: string; cls: string; icon: typeof CheckCircle2; spin?: boolean }> = {
  queued: { label: 'в очереди', cls: 'text-muted-foreground', icon: CircleDashed },
  running: { label: 'выполняется', cls: 'text-brand', icon: Loader2, spin: true },
  done: { label: 'готово', cls: 'text-good', icon: CheckCircle2 },
  failed: { label: 'ошибка', cls: 'text-bad', icon: XCircle },
  cancelled: { label: 'отменено', cls: 'text-muted-foreground', icon: CircleSlash },
  absent: { label: 'не скачан', cls: 'text-muted-foreground', icon: Download },
  downloading: { label: 'скачивается', cls: 'text-brand', icon: Loader2, spin: true },
  downloaded: { label: 'скачан', cls: 'text-muted-foreground', icon: CircleDashed },
  preparing: { label: 'подготовка', cls: 'text-brand', icon: Loader2, spin: true },
  ready: { label: 'готов', cls: 'text-good', icon: CheckCircle2 },
  error: { label: 'ошибка', cls: 'text-bad', icon: XCircle },
}

export function StatusBadge({ status, className }: { status: Status; className?: string }) {
  const s = STATUS[status] ?? STATUS.queued
  const Icon = s.icon
  return (
    <Badge variant="outline" className={cn('gap-1 font-normal', s.cls, className)}>
      <Icon className={cn(s.spin && 'animate-spin')} aria-hidden />
      {s.label}
    </Badge>
  )
}

export function OverlapBadge({ overlap }: { overlap: DatasetOut['overlap'] }) {
  const label = { overlap: 'с перекрытием', nonoverlap: 'без перекрытия', mixed: 'смешанный' }[overlap]
  return (
    <Badge variant="secondary" className="font-normal">
      {label}
    </Badge>
  )
}

export function SplitBadge({ split, className }: { split: string; className?: string }) {
  return (
    <Badge variant="outline" className={cn('font-mono font-normal', split === 'test' && 'border-brand/40 text-brand', className)}>
      {split}
    </Badge>
  )
}
