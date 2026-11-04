"""Run from the repo root: python scripts/make_field.py

Generates templates/partials/field.svg: an original, monochrome, outline-style
pixel scene of a stadium (light tower, bleachers with a crowd, goalpost,
scoreboard) with a football in flight. Shapes are drawn as outlines: only the
edge cells of each shape are lit, which gives the line-art look.
"""
import random

W, H, CELL = 320, 96, 3
BRIGHT, MID, DIM = "#F2F2F2", "#7A7A7A", "#2A2A2A"
grid: dict[tuple[int, int], str] = {}


def px(x, y, c):
    if 0 <= x < W and 0 <= y < H:
        grid[(x, y)] = c


def clear(x, y, w, h):
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            grid.pop((xx, yy), None)


def outline_rect(x, y, w, h, c):
    for xx in range(x, x + w):
        px(xx, y, c); px(xx, y + h - 1, c)
    for yy in range(y, y + h):
        px(x, yy, c); px(x + w - 1, yy, c)


def outline_mask(rows, x0, y0, c):
    """Light only the edge cells of an ASCII mask ('#' = inside)."""
    cells = {(x, y) for y, r in enumerate(rows) for x, v in enumerate(r) if v == "#"}
    for (x, y) in cells:
        if any((x + dx, y + dy) not in cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            px(x0 + x, y0 + y, c)


DIGITS = {
    "0": ["###", "#.#", "#.#", "#.#", "###"], "1": [".#.", "##.", ".#.", ".#.", "###"],
    "2": ["###", "..#", "###", "#..", "###"], "3": ["###", "..#", "###", "..#", "###"],
    "4": ["#.#", "#.#", "###", "..#", "..#"], "7": ["###", "..#", ".#.", ".#.", ".#."],
}


def text(s, x0, y0, c, scale=1):
    for i, ch in enumerate(s):
        for dy, row in enumerate(DIGITS[ch]):
            for dx, v in enumerate(row):
                if v == "#":
                    for sy in range(scale):
                        for sx in range(scale):
                            px(x0 + (i * 4 + dx) * scale + sx, y0 + dy * scale + sy, c)


random.seed(11)
GROUND = 84

# --- sky: a few faint stars and two outlined clouds ---
for _ in range(22):
    px(random.randrange(W), random.randrange(2, 40), DIM)
CLOUD = [
    "......####...........",
    "....########..####...",
    "...##############....",
    "..#################..",
    ".###################.",
    "#####################",
    ".###################.",
]
SMALL_CLOUD = [
    "...####......",
    ".########.##.",
    "#############",
    ".###########.",
]
outline_mask(CLOUD, 108, 6, BRIGHT)
outline_mask(SMALL_CLOUD, 196, 4, BRIGHT)

# --- ground: tiled turf strip, yard ticks on the top edge ---
for x in range(W):
    px(x, GROUND, BRIGHT)
for row, y in enumerate(range(GROUND + 1, H, 6)):
    off = 0 if row % 2 == 0 else 4
    for x in range(-off, W, 8):
        outline_rect(x, y, 9, 7, MID)
for x in range(124, 232, 18):
    for y in range(GROUND - 2, GROUND):
        px(x, y, BRIGHT)

# --- light tower ---
for y in range(16, GROUND):
    px(9, y, BRIGHT); px(10, y, BRIGHT)
for y in range(20, GROUND, 6):  # lattice
    px(8, y, MID); px(11, y, MID)
outline_rect(3, 6, 14, 10, BRIGHT)
for lx in (6, 9, 12):
    for ly in (9, 12):
        px(lx, ly, BRIGHT); px(lx + 1, ly, BRIGHT)

# --- grandstand: seat rows rise away from the field (to the left) under a roof ---
BX, B, ROWS = 24, 10, 6
for col in range(ROWS):
    for level in range(ROWS - col):
        x, y = BX + col * B, GROUND - (level + 1) * B
        outline_rect(x, y, B + 1, B + 1, BRIGHT)
        if random.random() < 0.7:  # a fan: head + shoulders
            px(x + 4, y + 3, MID); px(x + 5, y + 3, MID)
            px(x + 4, y + 4, MID); px(x + 5, y + 4, MID)
            for dx in range(3, 7):
                px(x + dx, y + 6, MID)
            for dx in range(2, 8):
                px(x + dx, y + 7, MID)
        else:  # empty seat
            for dx in range(3, 7):
                px(x + dx, y + 7, DIM)
# back wall behind the top row, roof cantilevered toward the field
wall_x = BX - 2
top = GROUND - ROWS * B
roof_y0, roof_x0, roof_x1 = top - 14, wall_x, BX + ROWS * B + 8
for y in range(roof_y0, GROUND):
    px(wall_x, y, BRIGHT); px(wall_x + 1, y, BRIGHT)
def roof_y(x):
    return roof_y0 + round((x - roof_x0) * 0.12)
for x in range(roof_x0, roof_x1):
    px(x, roof_y(x), BRIGHT); px(x, roof_y(x) + 1, BRIGHT)
for x in range(roof_x0 + 6, roof_x1 - 2, 12):  # truss ticks under the roof
    for k in range(2, 5):
        px(x, roof_y(x) + k, MID)
# slim columns holding the roof's front edge, in front of the low rows
for cx in (BX + ROWS * B + 4,):
    for y in range(roof_y(cx) + 2, GROUND):
        if (cx, y) not in grid:
            px(cx, y, MID)

# --- goalpost (front view) ---
GX, BAR = 262, 54
for y in range(BAR, GROUND):
    px(GX, y, BRIGHT); px(GX + 1, y, BRIGHT)
for x in range(GX - 15, GX + 17):
    px(x, BAR, BRIGHT); px(x, BAR + 1, BRIGHT)
for y in range(12, BAR):
    px(GX - 15, y, BRIGHT); px(GX + 16, y, BRIGHT)

# --- scoreboard on legs ---
SX, SY, SW, SH = 286, 22, 30, 22
outline_rect(SX, SY, SW, SH, BRIGHT)
outline_rect(SX + 2, SY + 2, SW - 4, SH - 4, MID)
text("21", SX + 5, SY + 6, BRIGHT)
text("17", SX + 17, SY + 6, BRIGHT)
for dx in (5, 13, 21):
    px(SX + dx, SY + 15, MID); px(SX + dx + 1, SY + 15, MID)
for y in range(SY + SH, GROUND):
    px(SX + 7, y, BRIGHT); px(SX + SW - 8, y, BRIGHT)

# --- tee and ball flight: parabola from the tee to between the uprights ---
x0, y0, x1, y1, h = 132, GROUND - 2, GX, 30, 34
def ypath(x):
    t = (x - x0) / (x1 - x0)
    return round(y0 + (y1 - y0) * t - 4 * h * t * (1 - t))
for dx in (-2, -1, 0, 1, 2):
    px(x0 + dx, GROUND - 1, BRIGHT)
px(x0, GROUND - 2, BRIGHT)
BALL_X = 208
for x in range(x0 + 3, BALL_X - 14, 4):        # flown: dots
    px(x, ypath(x), MID)
for x in range(BALL_X + 11, x1 - 10, 6):       # still to fly: faint dashes
    px(x, ypath(x), DIM); px(x + 1, ypath(x + 1), DIM)

BALL = [
    "....#######....",
    "..###########..",
    ".#############.",
    "###############",
    ".#############.",
    "..###########..",
    "....#######....",
]
by = ypath(BALL_X)
clear(BALL_X - 7, by - 3, 15, 7)
outline_mask(BALL, BALL_X - 7, by - 3, BRIGHT)
for dx in range(-3, 4):                         # lace line
    px(BALL_X + dx, by, BRIGHT)
for dx in (-2, 0, 2):                           # stitches
    px(BALL_X + dx, by - 1, BRIGHT); px(BALL_X + dx, by + 1, BRIGHT)
for dy, ln in ((-2, 4), (0, 6), (2, 4)):        # speed lines
    for k in range(ln):
        px(BALL_X - 10 - k, by + dy, MID)

# --- render: merge horizontal runs into rects ---
rects = []
for y in range(H):
    x = 0
    while x < W:
        c = grid.get((x, y))
        if c is None:
            x += 1
            continue
        start = x
        while x < W and grid.get((x, y)) == c:
            x += 1
        rects.append(f'<rect x="{start * CELL}" y="{y * CELL}" width="{(x - start) * CELL}" height="{CELL}" fill="{c}"/>')

svg = (
    f'<svg class="pixel-scene" viewBox="0 0 {W * CELL} {H * CELL}" xmlns="http://www.w3.org/2000/svg" '
    'shape-rendering="crispEdges" role="img" aria-label="Pixel art: a stadium with a light tower, '
    'bleachers full of fans, a goalpost and a scoreboard, with a football in flight">'
    + "".join(rects) + "</svg>\n"
)
with open("templates/partials/field.svg", "w") as f:
    f.write(svg)
print(len(rects), "rects,", len(svg), "bytes")
