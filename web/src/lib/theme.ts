import { createContext, useContext } from 'react'

export type Theme = 'light' | 'dark' | 'system'
export const THEME_KEY = 'mtmct-theme'

export interface ThemeCtx {
  theme: Theme
  resolvedTheme: 'light' | 'dark'
  setTheme: (t: Theme) => void
}

export const ThemeContext = createContext<ThemeCtx>({ theme: 'system', resolvedTheme: 'light', setTheme: () => {} })

export const useTheme = () => useContext(ThemeContext)
