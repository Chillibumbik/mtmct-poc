import { createRootRoute, createRoute, createRouter, lazyRouteComponent } from '@tanstack/react-router'
import { z } from 'zod'
import { AppShell } from '@/components/app/AppShell'
import { EmptyState, ErrorState } from '@/components/app/States'
import { DataPage } from '@/pages/Data'
import { ExperimentsPage } from '@/pages/Experiments'
import { JobsPage } from '@/pages/Jobs'
import { NewRunPage } from '@/pages/NewRun'
import { OverviewPage } from '@/pages/Overview'
import { PlaceholderPage } from '@/pages/Placeholder'

// Search params are part of the URL so every view is linkable (spec §10.2 UX).
// TanStack Router JSON-parses search values, hence the coercions.
const str = z.coerce.string().optional().catch(undefined)
const num = z.coerce.number().int().optional().catch(undefined)

const rootRoute = createRootRoute({
  component: AppShell,
  errorComponent: ({ error, reset }) => <ErrorState error={error} onRetry={reset} />,
  notFoundComponent: () => <EmptyState title="Страница не найдена" hint="Проверьте адрес или выберите раздел в меню." />,
})

const overview = createRoute({ getParentRoute: () => rootRoute, path: '/', component: OverviewPage })
const data = createRoute({ getParentRoute: () => rootRoute, path: '/data', component: DataPage })

export const newRunSearch = z.object({
  tab: z.enum(['single', 'sweep']).optional().catch(undefined),
  dataset: str,
  sequence: str,
  split: str,
  mode: z.enum(['full', 'oracle']).optional().catch(undefined),
})
const newRun = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs/new',
  validateSearch: (s) => newRunSearch.parse(s),
  component: NewRunPage,
})

const jobs = createRoute({
  getParentRoute: () => rootRoute,
  path: '/jobs',
  validateSearch: (s) => z.object({ job: num }).parse(s),
  component: JobsPage,
})

export const experimentsSearch = z.object({
  dataset: str,
  split: str,
  mode: str,
  status: str,
  sweep: num,
  q: str,
  sort: str,
  desc: z.coerce.boolean().optional().catch(undefined),
})
const experiments = createRoute({
  getParentRoute: () => rootRoute,
  path: '/experiments',
  validateSearch: (s) => experimentsSearch.parse(s),
  component: ExperimentsPage,
})

const runDetail = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs/$runId',
  // ECharts pages are split into their own chunks.
  component: lazyRouteComponent(() => import('@/pages/RunDetail'), 'RunDetailPage'),
})

// Built by T7 (web/src/features/{player,errors,compare}); stubs keep links and URLs stable.
const runPlayer = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs/$runId/player',
  validateSearch: (s) => z.object({ t: z.coerce.number().optional().catch(undefined), camera: str, gid: num }).parse(s),
  component: () => <PlaceholderPage title="Плеер" />,
})
const runErrors = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs/$runId/errors',
  component: () => <PlaceholderPage title="Разбор ошибок" />,
})
const compare = createRoute({
  getParentRoute: () => rootRoute,
  path: '/compare',
  validateSearch: (s) => z.object({ runs: str }).parse(s),
  component: () => <PlaceholderPage title="Сравнение прогонов" />,
})

const resources = createRoute({
  getParentRoute: () => rootRoute,
  path: '/resources',
  validateSearch: (s) => z.object({ dataset: str }).parse(s),
  component: lazyRouteComponent(() => import('@/pages/Resources'), 'ResourcesPage'),
})

const routeTree = rootRoute.addChildren([
  overview,
  data,
  newRun,
  jobs,
  experiments,
  runDetail,
  runPlayer,
  runErrors,
  compare,
  resources,
])

export const router = createRouter({ routeTree, defaultPreload: 'intent', scrollRestoration: true })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
