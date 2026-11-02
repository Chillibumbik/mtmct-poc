import { Monitor, Moon, Sun } from 'lucide-react'
import { useCallback, useEffect, useState, useSyncExternalStore, type ReactNode } from 'react'
import { THEME_KEY, ThemeContext, useTheme, type Theme } from '@/lib/theme'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

const NEXT = { system: 'light', light: 'dark', dark: 'system' } as const
const LABEL = { system: 'Тема: системная', light: 'Тема: светлая', dark: 'Тема: тёмная' } as const

export function ThemeToggle() {
  const { theme = 'system', setTheme } = useTheme()
  const key = (theme in NEXT ? theme : 'system') as keyof typeof NEXT
  const Icon = key === 'light' ? Sun : key === 'dark' ? Moon : Monitor
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label={LABEL[key]} onClick={() => setTheme(NEXT[key])}>
          <Icon />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{LABEL[key]}</TooltipContent>
    </Tooltip>
  )
}

const media = () => window.matchMedia('(prefers-color-scheme: dark)')
const subscribe = (cb: () => void) => {
  const m = media()
  m.addEventListener('change', cb)
  return () => m.removeEventListener('change', cb)
}

function stored(): Theme {
  try {
    const v = localStorage.getItem(THEME_KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

/** Light/dark/system theme on `<html class="dark">` (index.html applies it before first paint). */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(stored)
  const systemDark = useSyncExternalStore(subscribe, () => media().matches)
  const resolvedTheme = theme === 'system' ? (systemDark ? 'dark' : 'light') : theme
  useEffect(() => {
    document.documentElement.classList.toggle('dark', resolvedTheme === 'dark')
  }, [resolvedTheme])
  const setTheme = useCallback((t: Theme) => {
    setThemeState(t)
    try {
      if (t === 'system') localStorage.removeItem(THEME_KEY)
      else localStorage.setItem(THEME_KEY, t)
    } catch {
      // storage unavailable (private mode): theme just isn't remembered
    }
  }, [])
  return <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>{children}</ThemeContext.Provider>
}
