/** Dotted-key helpers mirroring `mtmct.config.apply_overrides` (spec §6.3). */

export type Json = Record<string, unknown>

const isObj = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v)

export function getPath(obj: unknown, dotted: string): unknown {
  let node = obj
  for (const part of dotted.split('.')) {
    if (!isObj(node)) return undefined
    node = node[part]
  }
  return node
}

/** `{"a.b.c": 1}` → nested assignment, creating missing objects; later keys win. */
export function applyOverrides(cfg: Json, overrides: Json): Json {
  const out = structuredClone(cfg)
  for (const [dotted, value] of Object.entries(overrides)) {
    const parts = dotted.split('.')
    let node = out
    for (const p of parts.slice(0, -1)) {
      if (!isObj(node[p])) node[p] = {}
      node = node[p] as Json
    }
    node[parts.at(-1)!] = structuredClone(value)
  }
  return out
}

/** Leaf paths of a nested config (arrays and nulls are leaves). */
export function leafKeys(obj: Json, prefix = ''): string[] {
  return Object.entries(obj).flatMap(([k, v]) => {
    const key = prefix ? `${prefix}.${k}` : k
    return isObj(v) && Object.keys(v).length > 0 ? leafKeys(v, key) : [key]
  })
}

export function deepEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

/** Parses one sweep grid cell: JSON literal when valid (0.5, true, null, "x", [1]), else a bare string. */
export function parseGridValue(text: string): unknown {
  const t = text.trim()
  try {
    return JSON.parse(t)
  } catch {
    return t
  }
}

/** "0.3, 0.45, yolox_tiny" → [0.3, 0.45, "yolox_tiny"]; empty cells dropped. */
export function parseGridValues(text: string): unknown[] {
  return text
    .split(/[,;]/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(parseGridValue)
}
