import { useSyncExternalStore } from 'react'
import { Pause, Play, SkipBack, StepBack, StepForward } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { SPEEDS, stepFrame, type MasterClock } from './clock'
import { formatClock } from './format'

/** Transport bar: play/pause, frame step, speed, scrubber on the common clock (with optional highlighted spans). */
export function Controls({ clock, frameS, marks }: { clock: MasterClock; frameS: number; marks: { from: number; to: number; color: string }[] }) {
  const s = useSyncExternalStore(clock.subscribe, clock.getSnapshot)
  const span = Math.max(1e-6, clock.max - clock.min)
  return (
    <div className="sticky bottom-0 z-10 -mx-6 mt-4 border-t bg-background/95 px-6 py-2.5 backdrop-blur lg:-mx-8 lg:px-8">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <div className="flex items-center gap-0.5">
          <Button size="icon-sm" variant="ghost" onClick={() => clock.seek(clock.min)} aria-label="В начало" title="В начало">
            <SkipBack />
          </Button>
          <Button size="icon-sm" variant="ghost" onClick={() => stepFrame(clock, -frameS)} aria-label="Кадр назад" title="Кадр назад (←)">
            <StepBack />
          </Button>
          <Button size="icon" onClick={() => clock.toggle()} aria-label={s.playing ? 'Пауза' : 'Воспроизвести'} title="Пробел">
            {s.playing ? <Pause /> : <Play />}
          </Button>
          <Button size="icon-sm" variant="ghost" onClick={() => stepFrame(clock, frameS)} aria-label="Кадр вперёд" title="Кадр вперёд (→)">
            <StepForward />
          </Button>
        </div>
        <div className="font-mono text-sm tabular-nums" aria-live="off">
          {formatClock(s.t)} <span className="text-muted-foreground">/ {formatClock(clock.max)}</span>
        </div>
        <ToggleGroup
          type="single"
          size="sm"
          variant="outline"
          value={String(s.speed)}
          onValueChange={(v) => v && clock.setSpeed(Number(v))}
          aria-label="Скорость воспроизведения"
        >
          {SPEEDS.map((v) => (
            <ToggleGroupItem key={v} value={String(v)} className="px-2 font-mono text-xs">
              {v}×
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <div className="ml-auto hidden text-xs text-muted-foreground xl:block">
          Пробел — пуск/пауза · ←/→ — кадр (Shift — 1 с) · [ ] — скорость · Esc — снять выбор
        </div>
      </div>
      <div className="relative mt-2 pb-1.5">
        <Slider
          min={clock.min}
          max={clock.max}
          step={0.01}
          value={[s.t]}
          onValueChange={([v]) => clock.seek(v)}
          aria-label="Время на общей шкале"
        />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-1">
          {marks.map((m, i) => (
            <div
              key={i}
              className="absolute inset-y-0 min-w-[2px] rounded-full"
              style={{ left: `${((m.from - clock.min) / span) * 100}%`, width: `${((m.to - m.from) / span) * 100}%`, background: m.color }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
