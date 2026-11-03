import type { CompareRow, RunOut } from '@/api/types.gen'
import { bestRuns, deltaTone, formatDelta, varyingKeys } from './compare'

const run = (id: number, config_diff: Record<string, unknown>) => ({ id, name: `r${id}`, config_diff }) as unknown as RunOut
const row = (values: Record<number, number | null>, best: number | null, higher = true) =>
  ({ metric: 'cross_idf1', values, deltas: {}, best_run_id: best, higher_is_better: higher }) as unknown as CompareRow

describe('compare helpers', () => {
  it('keeps only config keys that differ between runs', () => {
    const runs = [
      run(1, { 'data.max_seconds': 30, 'associate.solver.params.threshold': 0.3 }),
      run(2, { 'data.max_seconds': 30, 'associate.solver.params.threshold': 0.6 }),
      run(3, { 'data.max_seconds': 30, 'associate.solver.params.threshold': 0.3, 'refine.enabled': false }),
    ]
    expect(varyingKeys(runs)).toEqual(['associate.solver.params.threshold', 'refine.enabled'])
    expect(varyingKeys([runs[0], runs[0]])).toEqual([])
  })

  it('formats deltas in display units and colours them by metric direction', () => {
    expect(formatDelta('cross_idf1', 0.0321)).toBe('+3.2')
    expect(formatDelta('cross_idf1', -0.05)).toBe('−5.0')
    expect(formatDelta('false_merges', 2)).toBe('+2')
    expect(formatDelta('cross_idf1', 0)).toBe('±0')
    expect(formatDelta('cross_idf1', null)).toBe('')
    expect(deltaTone(0.1, true)).toBe('good')
    expect(deltaTone(2, false)).toBe('bad') // more false merges is worse
    expect(deltaTone(-0.5, false)).toBe('good') // less RAM is better
    expect(deltaTone(0, true)).toBe('neutral')
  })

  it('highlights every run holding the best value, and none when all are equal', () => {
    expect(bestRuns(row({ 1: 0.5, 2: 0.7 }, 2))).toEqual(new Set([2]))
    expect(bestRuns(row({ 1: 1, 2: 0.5, 3: 1 }, 1))).toEqual(new Set([1, 3]))
    expect(bestRuns(row({ 1: 3, 2: 0, 3: 0 }, 2, false))).toEqual(new Set([2, 3]))
    expect(bestRuns(row({ 1: 1, 2: 1 }, 1))).toEqual(new Set())
    expect(bestRuns(row({ 1: 0.4, 2: null }, 1))).toEqual(new Set())
  })
})
