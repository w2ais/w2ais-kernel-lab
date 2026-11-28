"""Output-tile coordinates for the Ampere-style NVFP4 CTA grid."""

import cutlass
import cutlass.cute as cute


@cute.jit
def swizzled_tile(bid, m_tiles, n_tiles, width: cutlass.Constexpr, row_bands: cutlass.Constexpr):
    if cutlass.const_expr(row_bands):
        outer_tiles, band_tiles = n_tiles, m_tiles
    else:
        outer_tiles, band_tiles = m_tiles, n_tiles

    band = bid // (width * outer_tiles)
    band_start = band * width
    actual_width = min(width, band_tiles - band_start)
    in_band = bid - band_start * outer_tiles
    outer = in_band // actual_width
    inner = band_start + in_band % actual_width
    if cutlass.const_expr(row_bands):
        bm, bn = inner, outer
    else:
        bm, bn = outer, inner
    return bm, bn
