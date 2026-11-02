import { useEffect, useMemo, useRef } from 'react'
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, MarkLineComponent, TooltipComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'
import { baseOption, useChartTheme, type ChartTheme } from '@/lib/chart'

echarts.use([LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, CanvasRenderer])

export function Chart({
  option,
  height = 280,
  ariaLabel,
}: {
  option: (t: ChartTheme) => EChartsOption
  height?: number
  ariaLabel: string
}) {
  const t = useChartTheme()
  const opt = useMemo(() => {
    const base = baseOption(t)
    const o = option(t)
    const merge = (a: unknown, b: unknown) => (Array.isArray(b) ? b.map((x) => ({ ...(a as object), ...x })) : { ...(a as object), ...(b as object) })
    return { ...base, ...o, xAxis: merge(base.xAxis, o.xAxis), yAxis: merge(base.yAxis, o.yAxis), tooltip: { ...(base.tooltip as object), ...(o.tooltip as object) } }
  }, [t, option])
  const ref = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)
  useEffect(() => {
    const el = ref.current!
    const c = echarts.init(el)
    chart.current = c
    const ro = new ResizeObserver(() => c.resize())
    ro.observe(el)
    return () => {
      ro.disconnect()
      c.dispose()
      chart.current = null
    }
  }, [])
  useEffect(() => {
    chart.current?.setOption(opt, { notMerge: true })
  }, [opt])
  return <div ref={ref} role="img" aria-label={ariaLabel} style={{ height, width: '100%' }} />
}
