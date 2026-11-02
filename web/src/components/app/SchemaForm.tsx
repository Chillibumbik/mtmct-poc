import { useState } from 'react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { fieldsFromSchema, parseFieldInput, type FormField } from '@/lib/schemaForm'
import { cn } from '@/lib/utils'

const show = (v: unknown) => (v === undefined || v === null ? '' : typeof v === 'string' ? v : JSON.stringify(v))

/** Text-backed input that commits a parsed value on blur / Enter and shows parse errors. */
function TextField({ field, value, changed, onChange }: { field: FormField; value: unknown; changed: boolean; onChange: (v: unknown) => void }) {
  const [draft, setDraft] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const commit = () => {
    if (draft === null) return
    const r = parseFieldInput(field, draft)
    if (r.ok) {
      setError(null)
      setDraft(null)
      onChange(r.value)
    } else setError(r.error)
  }
  const id = `f-${field.key}`
  return (
    <div className="space-y-1">
      <Label htmlFor={id} className={cn('text-xs font-normal text-muted-foreground', changed && 'text-brand')} title={field.description}>
        {field.label}
        {field.min !== undefined && field.max !== undefined && <span className="opacity-60"> · {field.min}…{field.max}</span>}
      </Label>
      <Input
        id={id}
        inputMode={field.kind === 'number' || field.kind === 'integer' ? 'decimal' : undefined}
        className={cn('h-7 font-mono text-xs', error && 'border-destructive')}
        value={draft ?? show(value)}
        placeholder={field.nullable ? 'null' : undefined}
        aria-invalid={!!error}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === 'Enter' && commit()}
      />
      {error && <div className="text-[11px] text-destructive">{error}</div>}
    </div>
  )
}

function Field({ field, value, changed, onChange }: { field: FormField; value: unknown; changed: boolean; onChange: (v: unknown) => void }) {
  const labelCls = cn('text-xs font-normal text-muted-foreground', changed && 'text-brand')
  if (field.kind === 'boolean')
    return (
      <div className="flex h-full items-end pb-1.5">
        <label className="flex items-center gap-2" title={field.description}>
          <Switch checked={Boolean(value)} onCheckedChange={onChange} />
          <span className={labelCls}>{field.label}</span>
        </label>
      </div>
    )
  if (field.kind === 'enum') {
    const options = field.options ?? []
    const idx = options.findIndex((o) => o === value)
    return (
      <div className="space-y-1">
        <Label className={labelCls} title={field.description}>
          {field.label}
        </Label>
        <Select value={idx < 0 ? undefined : String(idx)} onValueChange={(i) => onChange(options[Number(i)])}>
          <SelectTrigger size="sm" className="w-full font-mono text-xs">
            <SelectValue placeholder="—" />
          </SelectTrigger>
          <SelectContent>
            {options.map((o, i) => (
              <SelectItem key={String(o)} value={String(i)} className="font-mono text-xs">
                {String(o)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    )
  }
  return <TextField field={field} value={value} changed={changed} onChange={onChange} />
}

/** Auto-generated form for a component's pydantic `Params` JSON schema. */
export function SchemaForm({
  schema,
  values,
  defaults,
  exclude = [],
  onChange,
}: {
  schema: Record<string, unknown>
  values: Record<string, unknown>
  defaults?: Record<string, unknown>
  exclude?: string[]
  onChange: (key: string, value: unknown) => void
}) {
  const fields = fieldsFromSchema(schema).filter((f) => !exclude.includes(f.key))
  if (fields.length === 0) return <div className="text-xs text-muted-foreground">Нет параметров</div>
  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-2.5 md:grid-cols-3">
      {fields.map((f) => {
        const value = f.key in values ? values[f.key] : f.default
        const base = defaults && f.key in defaults ? defaults[f.key] : f.default
        return <Field key={f.key} field={f} value={value} changed={JSON.stringify(value) !== JSON.stringify(base)} onChange={(v) => onChange(f.key, v)} />
      })}
    </div>
  )
}
