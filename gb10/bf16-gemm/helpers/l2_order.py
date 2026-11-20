"""Transport-independent group-N tile ordering, shared by both new arms."""
import cutlass
import cutlass.cute as cute

@cute.jit
def group_n_tile(linear, m_tiles, n_tiles, group: cutlass.Constexpr = 4):
    group_id = linear // (group * m_tiles)
    first_n = group_id * group
    width = min(group, n_tiles - first_n)
    in_group = linear - group_id * group * m_tiles
    return in_group // width, first_n + in_group % width
