"""
Grid coordinate system and block manipulation utilities.
9x9 grid in x-z plane with stacking in y-axis.
"""

VALID_X = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
VALID_Z = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
VALID_Y = [50, 150, 250, 350, 450]
VALID_COLORS = {
    "RED", "BLUE", "GREEN", "YELLOW", "PURPLE",
    "ORANGE", "WHITE", "BLACK", "BROWN", "PINK",
    "GREY", "GRAY", "CYAN",
}

GRID_SIZE = 9
CELL_SIZE = 100
GROUND_Y = 50
STACK_HEIGHT = 100


def parse_blocks(block_string: str) -> list[tuple[str, int, int, int]]:
    """Parse a semicolon-separated block string into list of (Color, x, y, z) tuples."""
    blocks = []
    if not block_string or not block_string.strip():
        return blocks

    for part in block_string.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(",")
        if len(tokens) == 4:
            color = tokens[0].strip().capitalize()
            try:
                x, y, z = int(tokens[1].strip()), int(tokens[2].strip()), int(tokens[3].strip())
                blocks.append((color, x, y, z))
            except ValueError:
                continue
    return blocks


def blocks_to_string(blocks: list[tuple[str, int, int, int]]) -> str:
    """Convert block list to semicolon-separated string."""
    return ";".join(f"{color},{x},{y},{z}" for color, x, y, z in blocks)


def normalize_block_set(blocks: list[tuple[str, int, int, int]]) -> set[str]:
    """Convert blocks to normalized set for comparison."""
    return {f"{c},{x},{y},{z}" for c, x, y, z in blocks}


def snap_to_grid(val: int, valid: list[int]) -> int:
    """Snap a value to the nearest valid grid position."""
    return min(valid, key=lambda v: abs(v - val))


def get_corner_positions() -> list[tuple[int, int]]:
    """Return (x, z) positions of the four corners."""
    return [(-400, -400), (-400, 400), (400, -400), (400, 400)]


def get_edge_positions() -> list[tuple[int, int]]:
    """Return (x, z) positions along the grid edges (excluding corners)."""
    edges = []
    for v in VALID_X:
        if v not in (-400, 400):
            edges.append((v, -400))
            edges.append((v, 400))
    for v in VALID_Z:
        if v not in (-400, 400):
            edges.append((-400, v))
            edges.append((400, v))
    return edges


def get_center_position() -> tuple[int, int]:
    """Return the center (origin) position."""
    return (0, 0)


def get_next_y(blocks: list[tuple[str, int, int, int]], x: int, z: int) -> int:
    """Get the next available y position at (x, z) given existing blocks."""
    occupied_y = [b[2] for b in blocks if b[1] == x and b[3] == z]
    if not occupied_y:
        return GROUND_Y
    return max(occupied_y) + STACK_HEIGHT
