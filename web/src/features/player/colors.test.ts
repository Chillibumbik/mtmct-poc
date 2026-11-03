import { ID_PALETTE, colorForId, textColorOn } from './colors'

describe('colorForId', () => {
  it('is stable across calls and pinned to known values', () => {
    expect(colorForId(1)).toBe(colorForId(1))
    // Pinned: changing the hash would silently recolour every saved screenshot / mental map.
    expect([0, 1, 2, 3, 42, 1000].map(colorForId)).toMatchSnapshot()
  })

  it('always returns a palette colour, also for negative and large ids', () => {
    for (const id of [-5, 0, 7, 2 ** 31 - 1, 123456789]) expect(ID_PALETTE).toContain(colorForId(id))
  })

  it('gives consecutive ids (typical cluster labels 0..n) different colours', () => {
    for (const start of [0, 3, 17, 1000]) {
      const used = new Set(Array.from({ length: ID_PALETTE.length }, (_, i) => colorForId(start + i)))
      expect(used.size).toBe(ID_PALETTE.length)
    }
  })

  it('picks readable label text', () => {
    expect(textColorOn('#F0E442')).toBe('#000')
    expect(textColorOn('#0072B2')).toBe('#fff')
  })
})
