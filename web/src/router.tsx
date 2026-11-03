import { createRootRoute, createRoute, createRouter, lazyRouteComponent } from '@tanstack/react-router'
import { z } from 'zod'
import { AppShell } from '@/components/app/AppShell'
import { EmptyState, ErrorState } from '@/components/app/States'
import { DataPage } from '@/pages/Data'
import { ExperimentsPage } from '@/pages/Experiments'
import { JobsPage } from '@/pages/Jobs'
import { NewRunPage } from '@/pages/NewRun'
import { OverviewPage } from '@/pages/Overview'

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

// T7 features (web/src/features/{player,bev,errors,compare}); lazy chunks.
export const playerSearch = z.object({
  t: z.coerce.number().optional().catch(undefined),
  camera: str,
  gid: num,
  mode: z.enum(['pred', 'gt', 'diff']).optional().catch(undefined),
  tab: z.enum(['grid', 'bev']).optional().catch(undefined),
})
const runPlayer = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs/$runId/player',
  validateSearch: (s) => playerSearch.parse(s),
  component: lazyRouteComponent(() => import('@/features/player/PlayerPage'), 'PlayerPage'),
})
const runErrors = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs/$runId/errors',
  validateSearch: (s) => z.object({ kind: z.enum(['merges', 'splits']).optional().catch(undefined) }).parse(s),
  component: lazyRouteComponent(() => import('@/features/errors/ErrorsPage'), 'ErrorsPage'),
})
const compare = createRoute({
  getParentRoute: () => rootRoute,
  path: '/compare',
  validateSearch: (s) => z.object({ runs: str }).parse(s),
  component: lazyRouteComponent(() => import('@/features/compare/ComparePage'), 'ComparePage'),
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
