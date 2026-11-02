import { describe, expect, it } from 'vitest'
import { applyOverrides, getPath, leafKeys, parseGridValues } from './config'

describe('applyOverrides', () => {
  it('assigns dotted keys like the backend and does not mutate input', () => {
    const cfg = { associate: { solver: { name: 'agglomerative', params: { threshold: 0.45 } } } }
    const out = applyOverrides(cfg, { 'associate.solver.params.threshold': 0.5, 'data.max_seconds': 30 })
    expect(getPath(out, 'associate.solver.params.threshold')).toBe(0.5)
    expect(getPath(out, 'data.max_seconds')).toBe(30)
    expect(cfg.associate.solver.params.threshold).toBe(0.45)
  })
  it('replaces whole sub-dicts when a parent key is overridden', () => {
    const out = applyOverrides({ detect: { name: 'yolox_s', params: { conf: 0.3, nms: 0.45 } } }, { 'detect.params': { conf: 0.2 } })
    expect(getPath(out, 'detect.params')).toEqual({ conf: 0.2 })
  })
})

describe('leafKeys', () => {
  it('lists dotted leaf paths', () => {
    expect(leafKeys({ a: { b: 1, c: { d: null } }, e: [1], f: {} })).toEqual(['a.b', 'a.c.d', 'e', 'f'])
  })
})

describe('parseGridValues', () => {
  it('parses JSON literals and bare strings', () => {
    expect(parseGridValues('0.3, 0.45 ,true, yolox_tiny, null')).toEqual([0.3, 0.45, true, 'yolox_tiny', null])
    expect(parseGridValues(' , ')).toEqual([])
  })
})
