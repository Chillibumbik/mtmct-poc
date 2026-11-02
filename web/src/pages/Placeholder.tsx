import { Link, useParams } from '@tanstack/react-router'
import { Construction } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/app/States'
import { PageHeader } from '@/components/app/PageHeader'

/** Route stub for views delivered by T7 (player, error browser, comparison). */
export function PlaceholderPage({ title }: { title: string }) {
  const params = useParams({ strict: false }) as { runId?: string }
  return (
    <>
      <PageHeader title={title} description={params.runId ? `Прогон #${params.runId}` : undefined} />
      <EmptyState
        icon={<Construction className="size-6" />}
        title="Экран в разработке"
        hint="Этот экран появится в следующей итерации; адрес страницы уже постоянный."
        action={
          params.runId ? (
            <Button asChild variant="outline" size="sm">
              <Link to="/runs/$runId" params={{ runId: params.runId }}>
                К прогону
              </Link>
            </Button>
          ) : (
            <Button asChild variant="outline" size="sm">
              <Link to="/experiments">К экспериментам</Link>
            </Button>
          )
        }
      />
    </>
  )
}
