/**
 * Colour per global id: Okabe–Ito + Paul Tol colours (distinguishable under common colour-vision deficiencies;
 * near-duplicates like yellow/sand or sky/cyan left out — the label carries the id anyway),
 * chosen by a fixed hash of the id so a given id never changes colour.
 */
export const ID_PALETTE = [
  // Okabe–Ito (without black) + Paul Tol muted wine/olive/purple; ordered so that ids 0, 1, 2, … (index 7·id mod 10)
  // read orange, blue, yellow, wine, sky, vermillion, green, purple, olive, pink
  '#E69F00',
  '#882255',
  '#009E73',
  '#CC79A7',
  '#F0E442',
  '#D55E00',
  '#999933',
  '#0072B2',
  '#56B4E9',
  '#AA4499',
] as const

export function colorForId(id: number): string {
  // multiplicative hash: any run of 10 consecutive ids gets 10 different colours, neighbours far apart in the list
  const n = ID_PALETTE.length
  return ID_PALETTE[(((Math.trunc(id) * 7) % n) + n) % n]
}

/** Black or white text, whichever reads better on `hex` (relative luminance). */
export function textColorOn(hex: string): '#000' | '#fff' {
  const n = parseInt(hex.slice(1), 16)
  const lin = (c: number) => {
    const s = c / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  const l = 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255)
  return l > 0.22 ? '#000' : '#fff'
}
