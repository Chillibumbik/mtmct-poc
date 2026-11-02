import { describe, expect, it } from 'vitest'
import { fieldsFromSchema, parseFieldInput } from './schemaForm'

// Shapes exactly as pydantic v2 `Model.model_json_schema()` emits them.
const SCHEMA = {
  title: 'Params',
  type: 'object',
  $defs: { Linkage: { enum: ['average', 'complete', 'single'], title: 'Linkage', type: 'string' } },
  properties: {
    conf: { default: 0.3, maximum: 1, minimum: 0, title: 'Conf', type: 'number', description: 'Score threshold' },
    input_size: { default: 640, enum: [416, 640], title: 'Input Size', type: 'integer' },
    batch: { default: 32, exclusiveMinimum: 0, title: 'Batch', type: 'integer' },
    hard: { default: true, title: 'Hard', type: 'boolean' },
    backend: { default: 'openvino', enum: ['openvino', 'onnxruntime'], title: 'Backend', type: 'string' },
    linkage: { $ref: '#/$defs/Linkage', default: 'average' },
    device: { default: 'CPU', title: 'Device', type: 'string' },
    mode: { const: 'fixed', default: 'fixed', title: 'Mode', type: 'string' },
    gate_m: { anyOf: [{ type: 'number' }, { type: 'null' }], default: null, title: 'Gate M' },
    weights: { items: { type: 'number' }, title: 'Weights', type: 'array', default: [1, 2] },
  },
}

describe('fieldsFromSchema', () => {
  const fields = Object.fromEntries(fieldsFromSchema(SCHEMA).map((f) => [f.key, f]))

  it('keeps property order and titles', () => {
    expect(fieldsFromSchema(SCHEMA).map((f) => f.key)).toEqual(Object.keys(SCHEMA.properties))
    expect(fields.input_size.label).toBe('Input Size')
  })
  it('maps number with bounds', () => {
    expect(fields.conf).toMatchObject({ kind: 'number', default: 0.3, min: 0, max: 1, description: 'Score threshold' })
  })
  it('maps integer with exclusive minimum', () => {
    expect(fields.batch).toMatchObject({ kind: 'integer', default: 32, min: 1, step: 1 })
  })
  it('maps boolean', () => {
    expect(fields.hard).toMatchObject({ kind: 'boolean', default: true })
  })
  it('maps inline enums (string and integer) to select options', () => {
    expect(fields.backend).toMatchObject({ kind: 'enum', options: ['openvino', 'onnxruntime'] })
    expect(fields.input_size).toMatchObject({ kind: 'enum', options: [416, 640], default: 640 })
  })
  it('resolves $ref enums and const literals', () => {
    expect(fields.linkage).toMatchObject({ kind: 'enum', options: ['average', 'complete', 'single'], default: 'average' })
    expect(fields.mode).toMatchObject({ kind: 'enum', options: ['fixed'] })
  })
  it('maps plain string', () => {
    expect(fields.device).toMatchObject({ kind: 'string', default: 'CPU' })
  })
  it('unwraps Optional[...] into a nullable field', () => {
    expect(fields.gate_m).toMatchObject({ kind: 'number', nullable: true, default: null })
  })
  it('falls back to JSON editing for arrays and objects', () => {
    expect(fields.weights).toMatchObject({ kind: 'json', default: [1, 2] })
  })
  it('handles empty or missing schemas', () => {
    expect(fieldsFromSchema({})).toEqual([])
    expect(fieldsFromSchema({ type: 'object', properties: {} })).toEqual([])
  })
})

describe('parseFieldInput', () => {
  const [conf, , batch, , , , , , gate, weights] = fieldsFromSchema(SCHEMA)

  it('parses numbers with dot or comma and checks bounds', () => {
    expect(parseFieldInput(conf, '0,5')).toEqual({ ok: true, value: 0.5 })
    expect(parseFieldInput(conf, '1.5').ok).toBe(false)
    expect(parseFieldInput(conf, 'abc').ok).toBe(false)
  })
  it('rejects fractional integers', () => {
    expect(parseFieldInput(batch, '1.5').ok).toBe(false)
    expect(parseFieldInput(batch, '64')).toEqual({ ok: true, value: 64 })
  })
  it('maps empty input to null only for nullable fields', () => {
    expect(parseFieldInput(gate, '')).toEqual({ ok: true, value: null })
    expect(parseFieldInput(conf, '').ok).toBe(false)
  })
  it('parses JSON fields', () => {
    expect(parseFieldInput(weights, '[1, 3]')).toEqual({ ok: true, value: [1, 3] })
    expect(parseFieldInput(weights, '[1,').ok).toBe(false)
  })
})
