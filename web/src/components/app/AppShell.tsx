import { Link, Outlet } from '@tanstack/react-router'
import { Cpu, Database, FlaskConical, LayoutDashboard, ListTodo, PlayCircle, Waypoints } from 'lucide-react'
import { useJobs } from '@/api/hooks'
import { Badge } from '@/components/ui/badge'
import { ThemeToggle } from './theme'

const NAV = [
  { to: '/', label: 'Обзор', icon: LayoutDashboard, exact: true },
  { to: '/data', label: 'Данные', icon: Database },
  { to: '/runs/new', label: 'Новый прогон', icon: PlayCircle },
  { to: '/jobs', label: 'Задачи', icon: ListTodo },
  { to: '/experiments', label: 'Эксперименты', icon: FlaskConical },
  { to: '/resources', label: 'Ресурсы', icon: Cpu },
] as const

function ActiveJobs() {
  const { data } = useJobs()
  const n = data?.filter((j) => j.status === 'running' || j.status === 'queued').length ?? 0
  return n ? (
    <Badge variant="secondary" className="ml-auto h-4.5 min-w-4.5 px-1 tabular-nums">
      {n}
    </Badge>
  ) : null
}

export function AppShell() {
  return (
    <div className="flex min-h-svh">
      <aside className="sticky top-0 flex h-svh w-52 shrink-0 flex-col border-r bg-sidebar text-sidebar-foreground">
        <div className="flex h-12 items-center gap-2 px-4">
          <Waypoints className="size-4.5 text-brand" />
          <span className="font-semibold tracking-tight">MTMCT</span>
          <span className="text-xs text-muted-foreground">PoC</span>
        </div>
        <nav className="flex flex-1 flex-col gap-0.5 px-2 py-2" aria-label="Разделы">
          {NAV.map(({ to, label, icon: Icon, ...rest }) => (
            <Link
              key={to}
              to={to}
              activeOptions={{ exact: 'exact' in rest, includeSearch: false }}
              className="flex h-8 items-center gap-2.5 rounded-md px-2.5 text-sm text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground data-[status=active]:bg-sidebar-accent data-[status=active]:font-medium data-[status=active]:text-sidebar-accent-foreground"
            >
              <Icon className="size-4" />
              {label}
              {to === '/jobs' && <ActiveJobs />}
            </Link>
          ))}
        </nav>
        <div className="flex items-center justify-between border-t px-3 py-2">
          <span className="text-xs text-muted-foreground">127.0.0.1:8000</span>
          <ThemeToggle />
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-6 py-5 lg:px-8">
        <Outlet />
      </main>
    </div>
  )
}
