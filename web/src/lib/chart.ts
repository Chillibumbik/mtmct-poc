import { useTheme } from '@/lib/theme'
import type { EChartsOption } from 'echarts'

export interface ChartTheme {
  dark: boolean
  text: string
  muted: string
  grid: string
  surface: string
  /** Categorical palette (fixed order, colour-blind safe; validated reference instance). */
  palette: string[]
}

const LIGHT: ChartTheme = {
  dark: false,
  text: '#0b0b0b',
  muted: '#6b6a66',
  grid: '#ebeae6',
  surface: '#ffffff',
  palette: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
}
const DARK: ChartTheme = {
  dark: true,
  text: '#f0efec',
  muted: '#a3a29a',
  grid: '#2c2c2a',
  surface: '#1c1c1c',
  palette: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
}

export function useChartTheme(): ChartTheme {
  const { resolvedTheme } = useTheme()
  return resolvedTheme === 'dark' ? DARK : LIGHT
}

/** Shared axis/tooltip styling so every chart reads as one system. */
export function baseOption(t: ChartTheme): EChartsOption {
  const axis = {
    axisLine: { lineStyle: { color: t.grid } },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: t.grid } },
    nameTextStyle: { color: t.muted, fontSize: 11 },
  }
  return {
    color: t.palette,
    backgroundColor: 'transparent',
    textStyle: { fontFamily: 'Geist Variable, sans-serif', color: t.text },
    grid: { left: 8, right: 16, top: 36, bottom: 8, containLabel: true },
    legend: { top: 0, left: 0, icon: 'roundRect', itemWidth: 10, itemHeight: 10, textStyle: { color: t.muted, fontSize: 12 } },
    tooltip: {
      backgroundColor: t.surface,
      borderColor: t.grid,
      textStyle: { color: t.text, fontSize: 12 },
      extraCssText: 'box-shadow: 0 4px 16px rgb(0 0 0 / 0.12); border-radius: 6px;',
    },
    xAxis: axis,
    yAxis: axis,
  }
}
