/** Maps a pydantic v2 JSON schema (component `Params`) to flat form fields. */

export type FieldKind = 'number' | 'integer' | 'boolean' | 'enum' | 'string' | 'json'
export type EnumOption = string | number | boolean

export interface FormField {
  key: string
  label: string
  kind: FieldKind
  default: unknown
  nullable: boolean
  description?: string
  min?: number
  max?: number
  step?: number
  options?: EnumOption[]
}

type JsonSchema = Record<string, unknown>

const isObj = (v: unknown): v is JsonSchema => typeof v === 'object' && v !== null && !Array.isArray(v)

function resolveRef(node: JsonSchema, root: JsonSchema): JsonSchema {
  const ref = node.$ref
  if (typeof ref !== 'string' || !ref.startsWith('#/')) return node
  let target: unknown = root
  for (const part of ref.slice(2).split('/')) target = isObj(target) ? target[part] : undefined
  // Sibling keywords (default, description) on the referencing node win.
  return isObj(target) ? { ...target, ...node, $ref: undefined } : node
}

function unwrapNullable(node: JsonSchema, root: JsonSchema): { node: JsonSchema; nullable: boolean } {
  const variants = (node.anyOf ?? node.oneOf) as unknown
  if (!Array.isArray(variants)) return { node: resolveRef(node, root), nullable: false }
  const nonNull = variants.filter((v) => !(isObj(v) && v.type === 'null'))
  const nullable = nonNull.length < variants.length
  if (nonNull.length !== 1 || !isObj(nonNull[0])) return { node: { ...node, type: 'json' }, nullable }
  const { anyOf: _a, oneOf: _o, ...rest } = node
  return { node: { ...resolveRef(nonNull[0], root), ...rest }, nullable }
}

function fieldFor(key: string, raw: JsonSchema, root: JsonSchema): FormField {
  const { node, nullable } = unwrapNullable(raw, root)
  const base: FormField = {
    key,
    label: typeof node.title === 'string' ? node.title : key,
    kind: 'json',
    default: node.default ?? (nullable ? null : undefined),
    nullable,
    description: typeof node.description === 'string' ? node.description : undefined,
  }
  if (Array.isArray(node.enum)) return { ...base, kind: 'enum', options: node.enum as EnumOption[] }
  if (node.const !== undefined) return { ...base, kind: 'enum', options: [node.const as EnumOption] }

  const num = (k: string) => (typeof node[k] === 'number' ? (node[k] as number) : undefined)
  switch (node.type) {
    case 'boolean':
      return { ...base, kind: 'boolean' }
    case 'string':
      return { ...base, kind: 'string' }
    case 'integer':
    case 'number': {
      const integer = node.type === 'integer'
      const eps = integer ? 1 : 0
      const exMin = num('exclusiveMinimum')
      const exMax = num('exclusiveMaximum')
      return {
        ...base,
        kind: integer ? 'integer' : 'number',
        min: num('minimum') ?? (exMin === undefined ? undefined : exMin + eps),
        max: num('maximum') ?? (exMax === undefined ? undefined : exMax - eps),
        step: integer ? 1 : undefined,
      }
    }
    default:
      return base
  }
}

export function fieldsFromSchema(schema: JsonSchema): FormField[] {
  const props = schema.properties
  if (!isObj(props)) return []
  return Object.entries(props)
    .filter((entry): entry is [string, JsonSchema] => isObj(entry[1]))
    .map(([key, prop]) => fieldFor(key, prop, schema))
}

export type ParseResult = { ok: true; value: unknown } | { ok: false; error: string }

/** Parses text typed into a number/integer/json/string field. */
export function parseFieldInput(field: FormField, text: string): ParseResult {
  const t = text.trim()
  if (t === '') return field.nullable ? { ok: true, value: null } : { ok: false, error: 'Обязательное поле' }
  switch (field.kind) {
    case 'number':
    case 'integer': {
      const v = Number(t.replace(',', '.'))
      if (!Number.isFinite(v)) return { ok: false, error: 'Нужно число' }
      if (field.kind === 'integer' && !Number.isInteger(v)) return { ok: false, error: 'Нужно целое число' }
      if (field.min !== undefined && v < field.min) return { ok: false, error: `Минимум ${field.min}` }
      if (field.max !== undefined && v > field.max) return { ok: false, error: `Максимум ${field.max}` }
      return { ok: true, value: v }
    }
    case 'json':
      try {
        return { ok: true, value: JSON.parse(t) }
      } catch {
        return { ok: false, error: 'Некорректный JSON' }
      }
    default:
      return { ok: true, value: text }
  }
}
