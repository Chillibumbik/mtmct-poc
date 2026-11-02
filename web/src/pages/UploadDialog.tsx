import { useRef, useState, type DragEvent } from 'react'
import { FileJson, FileVideo, Upload, X } from 'lucide-react'
import { toast } from 'sonner'
import { errorText } from '@/api/client'
import { useUploadCustom } from '@/api/hooks'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

const VIDEO_RE = /\.(mp4|avi|mkv|mov|webm)$/i
const mb = (b: number) => `${(b / 1024 / 1024).toFixed(1)} МБ`

function OptionalFile({ id, label, hint, accept, file, onChange }: { id: string; label: string; hint: string; accept: string; file: File | null; onChange: (f: File | null) => void }) {
  return (
    <div className="space-y-1">
      <div className="text-sm font-medium">{label}</div>
      {file ? (
        <div className="flex items-center gap-2 rounded-md border px-2 py-1.5 text-sm">
          <FileJson className="size-4 text-muted-foreground" />
          <span className="truncate">{file.name}</span>
          <Button variant="ghost" size="icon-xs" className="ml-auto" aria-label="Убрать файл" onClick={() => onChange(null)}>
            <X />
          </Button>
        </div>
      ) : (
        <label className="flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-dashed px-2.5 text-sm text-muted-foreground hover:bg-muted/50">
          <Upload className="size-3.5" /> Выбрать файл…
          <input id={id} type="file" accept={accept} className="sr-only" onChange={(e) => onChange(e.target.files?.[0] ?? null)} />
        </label>
      )}
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  )
}

export function UploadDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const [name, setName] = useState('')
  const [videos, setVideos] = useState<File[]>([])
  const [calibration, setCalibration] = useState<File | null>(null)
  const [gt, setGt] = useState<File | null>(null)
  const [drag, setDrag] = useState(false)
  const [progress, setProgress] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  const upload = useUploadCustom(setProgress)

  const addFiles = (files: FileList | File[]) => {
    const list = [...files]
    const bad = list.filter((f) => !VIDEO_RE.test(f.name))
    if (bad.length) toast.error(`Не видео: ${bad.map((f) => f.name).join(', ')}`)
    const ok = list.filter((f) => VIDEO_RE.test(f.name))
    setVideos((v) => [...v, ...ok.filter((f) => !v.some((x) => x.name === f.name))])
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDrag(false)
    addFiles(e.dataTransfer.files)
  }
  const reset = () => {
    setName('')
    setVideos([])
    setCalibration(null)
    setGt(null)
    setProgress(0)
    upload.reset()
  }
  const submit = () =>
    upload.mutate(
      { name: name.trim(), videos, calibration, gt },
      {
        onSuccess: (job) => {
          toast.success(`Набор загружен, подготовка — задача #${job.id}`)
          reset()
          onOpenChange(false)
        },
      },
    )

  const valid = name.trim().length > 0 && videos.length > 0
  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (!upload.isPending) onOpenChange(v)
      }}
    >
      <DialogTrigger asChild>
        <Button size="sm" variant="outline">
          <Upload /> Загрузить свои видео
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Собственный набор камер</DialogTitle>
          <DialogDescription>Видео синхронных камер (одно на камеру). Калибровка и разметка GT — по желанию: без GT метрики качества не считаются.</DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-1">
            <Label htmlFor="ds-name">Название</Label>
            <Input id="ds-name" value={name} placeholder="Например, office_entrance" onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="space-y-1.5">
            <Label>Видео камер</Label>
            <div
              role="button"
              tabIndex={0}
              onClick={() => input.current?.click()}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
              onDragOver={(e) => {
                e.preventDefault()
                setDrag(true)
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
              className={cn(
                'flex cursor-pointer flex-col items-center gap-1 rounded-lg border border-dashed px-4 py-6 text-center text-sm transition-colors',
                drag ? 'border-brand bg-brand/5' : 'hover:bg-muted/50',
              )}
            >
              <Upload className="size-5 text-muted-foreground" />
              <span>Перетащите файлы или нажмите, чтобы выбрать</span>
              <span className="text-xs text-muted-foreground">mp4, avi, mkv, mov, webm · порядок файлов = порядок камер</span>
              <input
                ref={input}
                type="file"
                multiple
                accept="video/*,.mp4,.avi,.mkv,.mov,.webm"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) addFiles(e.target.files)
                  e.target.value = ''
                }}
              />
            </div>
            {videos.length > 0 && (
              <ul className="divide-y rounded-md border text-sm">
                {videos.map((f, i) => (
                  <li key={f.name} className="flex items-center gap-2 px-2 py-1">
                    <span className="w-6 font-mono text-xs text-muted-foreground">c{i}</span>
                    <FileVideo className="size-4 text-muted-foreground" />
                    <span className="truncate">{f.name}</span>
                    <span className="ml-auto shrink-0 text-xs tabular-nums text-muted-foreground">{mb(f.size)}</span>
                    <Button variant="ghost" size="icon-xs" aria-label={`Убрать ${f.name}`} onClick={() => setVideos((v) => v.filter((x) => x !== f))}>
                      <X />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
            {videos.length === 1 && <p className="text-xs text-warn">Для межкамерного трекинга нужно хотя бы две камеры.</p>}
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <OptionalFile id="calib" label="Калибровка (опц.)" hint="JSON: гомографии image→world по камерам" accept=".json,application/json" file={calibration} onChange={setCalibration} />
            <OptionalFile id="gt" label="Разметка GT (опц.)" hint="MOT txt или CSV с camera_id" accept=".txt,.csv" file={gt} onChange={setGt} />
          </div>
          {upload.isPending && (
            <div className="space-y-1">
              <Progress value={progress * 100} className="h-1.5" />
              <div className="text-xs tabular-nums text-muted-foreground">Загрузка {Math.round(progress * 100)}%</div>
            </div>
          )}
          {upload.isError && <div className="text-sm text-bad">{errorText(upload.error)}</div>}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={upload.isPending}>
            Отмена
          </Button>
          <Button onClick={submit} disabled={!valid || upload.isPending}>
            Загрузить
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
