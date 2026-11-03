import { fitBounds, niceStep } from './bev'

describe('BEV transform', () => {
  it('fits bounds into the canvas keeping aspect, centred, y up', () => {
    const { toCanvas, scale } = fitBounds([0, 0, 10, 5], 220, 220, 10)
    expect(scale).toBe(20) // limited by width: 200 px / 10 m
    expect(toCanvas(0, 0)).toEqual([10, 160]) // bottom-left, vertically centred (60 px above and below)
    expect(toCanvas(10, 5)).toEqual([210, 60])
  })

  it('handles degenerate bounds (single point)', () => {
    const [x, y] = fitBounds([3, 3, 3, 3], 100, 100, 10).toCanvas(3, 3)
    expect(Number.isFinite(x) && Number.isFinite(y)).toBe(true)
  })

  it('picks 1/2/5 grid steps', () => {
    expect(niceStep(8)).toBe(1)
    expect(niceStep(12)).toBe(2)
    expect(niceStep(35)).toBe(5)
    expect(niceStep(70)).toBe(10)
  })
})
