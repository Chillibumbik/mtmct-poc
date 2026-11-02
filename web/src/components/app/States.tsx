import type { ReactNode } from 'react'
import { AlertTriangle, Inbox, RotateCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { errorText } from '@/api/client'
import { cn } from '@/lib/utils'

export function LoadingBlock({ rows = 3, className }: { rows?: number; className?: string }) {
  return (
    <div className={cn('space-y-2', className)} aria-busy="true" aria-label="Загрузка">
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-8 w-full" style={{ opacity: 1 - i * 0.15 }} />
      ))}
    </div>
  )
}

export function EmptyState({
  title,
  hint,
  action,
  icon,
  className,
}: {
  title: string
  hint?: ReactNode
  action?: ReactNode
  icon?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-10 text-center', className)}>
      <div className="text-muted-foreground">{icon ?? <Inbox className="size-6" />}</div>
      <div className="font-medium">{title}</div>
      {hint && <div className="max-w-md text-sm text-muted-foreground">{hint}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  )
}

export function ErrorState({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  return (
    <div role="alert" className={cn('flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm', className)}>
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
      <div className="min-w-0 flex-1">
        <div className="font-medium text-destructive">Не удалось загрузить данные</div>
        <div className="break-words text-muted-foreground">{errorText(error)}</div>
      </div>
      {onRetry && (
        <Button size="sm" variant="outline" onClick={onRetry}>
          <RotateCw /> Повторить
        </Button>
      )}
    </div>
  )
}

interface QueryLike<T> {
  data: T | undefined
  isPending: boolean
  isError: boolean
  error: unknown
  refetch: () => unknown
}

/** Renders loading / error / empty / data states of one query. */
export function QueryView<T>({
  query,
  children,
  empty,
  isEmpty,
  loading,
}: {
  query: QueryLike<T>
  children: (data: T) => ReactNode
  empty?: ReactNode
  isEmpty?: (data: T) => boolean
  loading?: ReactNode
}) {
  if (query.isPending) return <>{loading ?? <LoadingBlock />}</>
  if (query.isError) return <ErrorState error={query.error} onRetry={() => void query.refetch()} />
  const data = query.data as T
  if (empty && isEmpty?.(data)) return <>{empty}</>
  return <>{children(data)}</>
}
