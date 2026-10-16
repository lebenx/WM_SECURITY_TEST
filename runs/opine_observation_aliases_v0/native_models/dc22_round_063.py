# ARC-AGI-3 -- Frames-only world model (dc22)
# GOAL (CONFIRMED L1, observed step 131): deliver the GREEN token onto the
# YELLOW target token (rows20-21, cols24-25). The winning move (green at
# rows20-21 cols22-23, press RIGHT) lands green on yellow -> reward 1.0, done,
# level advances to L2. The board has TWO vertical climb corridors and one
# horizontal band strip; the full delivery path is: climb the LEFT corridor to
# the slot -> traverse the band strip RIGHT (needs red=H) to the darkred base ->
# toggle red=V to bridge rows24-29 -> make blue-A solid -> climb the RIGHT
# corridor up through red-V and blue-A to the yellow token frame -> step RIGHT
# onto the yellow token.
#
# (historical, now superseded) earlier low-confidence hypothesis:
# climb GREEN to the RIGHT-corridor SUMMIT --
# the BLUE_A region (rows20-23, cols18-21). The board has TWO vertical climb
# corridors and one horizontal band strip joining them:
#   LEFT corridor  (cols8-11):  frame rows38-43 -> blue blob rows34-37 -> slot
#                               rows30-33. Climb the blob only when SOLID.
#   BAND strip     (rows30-33): slot gray cols8-11 | red bar cols12-17 (present
#                               only when red=H) | darkred base cols18-21.
#   RIGHT corridor (cols18-21): darkred rows30-33 -> red-V rows24-29 (present/
#                               climbable only when red=V) -> blue-A rows20-23.
# To reach the right summit the player must: climb the left corridor to the slot,
# traverse the band strip RIGHT to the darkred base (needs red=H so the bar is
# floor), then ASSEMBLE red into V (so rows24-29 become climbable red), then
# climb the right corridor into blue-A.
#
# WHY THIS PREDICATE (low confidence -- no reward ever observed in 122 steps):
#   - The entire LEFT slot + band strip (cols8-21, rows30-33) is swept at 0
#     reward (runs 7-9), so left-side delivery is falsified.
#   - The RIGHT corridor (red-V rows24-29 + blue-A rows20-23) has NEVER been
#     entered by green; reaching it requires the deliberate puzzle act of
#     assembling red=V after carrying green to its foot -- a genuine precondition,
#     not just an unvisited cell. Buffer-safe AND reachable (planner: 16 steps).
#   MODELING DEBT: true win condition unconfirmed (0 reward in 122 steps). This is
#   a structurally-motivated guess; refine on first real reward.
#
# Observed mechanics (level 1):
#   - A 2x2 green (palette 14) token is a free-moving sprite over a static
#     COMPOSITE background. ACTION1/2/3/4 = up/down/left/right slide it 1 cell
#     (2 px). A move is allowed iff EVERY destination cell is a "floor" colour
#     {gray=2, blue=9, red=8, darkred=13}; otherwise blocked (no-op). Vacated
#     cells revert to their static fill, sampled along the region's uniform axis
#     (vertically within the band strip rows30-33; horizontally in the vertical
#     corridors). This single rule subsumes the old "frame clamp" (frame fenced
#     by panel-bg=4), the climb up the solid blob, AND the band-strip traverse
#     where green slides over the red bar and darkred base.
#   - ACTION6 click on the RIGHT-panel RED preview toggles the red(8) workspace
#     piece between H (rows30-33,cols12-17) and V (rows24-29,cols18-21).
#   - ACTION6 click on the RIGHT-panel BLUE preview toggles each of the two
#     blue(9) regions (A: rows20-23,cols18-21; B: rows34-37,cols8-11) between a
#     SOLID block and a CHECKER pattern ((r+c)%2==1 -> 9, else bg 4). The two
#     regions are always in opposite phase. Making blob B solid builds the bridge.
#   - The FIRST preview click also removes the one-time white(0) outlines from
#     BOTH previews, redrawing them as plain solid anvils (pure fn of the frame).
#   - Other ACTION6 clicks (left workspace) and ACTION7 (undo) are board no-ops.
#   - Row 63 is a HUD move-counter strip; excluded via move_counter_mask().

import copy

GREEN = 14   # player token colour
FRAME = 2    # gray frame colour
PLAYER_COLOR = GREEN
RED = 8
BLUE = 9
DARKRED = 13
YELLOW = 11   # the inert target token colour (goal: deliver green to it)
LEFT_BG = 4
RIGHT_BG = 5
RIGHT_X = 34   # first column of the RIGHT target/preview panel

# Action ids (verified from buffer: 1=up, 3=left, 2=down; 4=right by symmetry)
A_UP, A_DOWN, A_LEFT, A_RIGHT = 1, 2, 3, 4


class Sprite:
    """A 2D sprite. -1 marks transparent cells; higher layer draws on top."""

    def __init__(self, pixels, x=0, y=0, layer=0, name="", visible=True):
        self.pixels = pixels
        self.x = int(x)
        self.y = int(y)
        self.layer = int(layer)
        self.name = name
        self.visible = bool(visible)


def render_sprites(sprites, w=64, h=64, background=5):
    canvas = [[int(background)] * int(w) for _ in range(int(h))]
    for s in sorted([sp for sp in sprites if sp.visible], key=lambda s: s.layer):
        for dy, row in enumerate(s.pixels):
            ry = s.y + dy
            if ry < 0 or ry >= h:
                continue
            for dx, v in enumerate(row):
                rx = s.x + dx
                if rx < 0 or rx >= w:
                    continue
                v = int(v)
                if v < 0:
                    continue
                canvas[ry][rx] = v
    return canvas


def load_level_frame(fname):
    """Load a cached per-level entry frame (for RESET / level-advance)."""
    import os, pickle
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, fname), 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


# --- Geometry helpers --------------------------------------------------------

def _find_player(grid):
    """Return (top, left, bottom, right) bbox of the green token, or None.

    The player is the 2x2 green token on the LEFT workspace board. Later levels
    can also draw stray green cells in the RIGHT panel (anvil/button decoration);
    taking the bbox over ALL green cells would merge those into one giant box and
    break movement. So restrict to left-workspace green and pick the largest
    connected component (the token)."""
    comps = [k for k in _components(grid, GREEN) if k["bbox"][3] < RIGHT_X]
    if not comps:
        return None
    comps.sort(key=lambda k: k["size"])
    return comps[-1]["bbox"]


# Colours the green token can stand on / move onto. Everything else (panel
# background 4/5, border 3, checker gaps) blocks the move. The token slides over
# a STATIC composite background made of these region fills:
#   gray frame/slot=2, solid blue blob=9, red bar=8, darkred base=13.
# This subsumes the old frame clamp (the home frame is fenced by panel bg=4),
# the climb up the SOLID blue blob bridge (blue=9 is floor only where solid; in
# the checker phase the destination has bg=4 gaps and the move is blocked), AND
# the horizontal traverse along rows30-33 (gray slot -> red bar -> darkred base).
_FLOOR = (FRAME, BLUE, RED, DARKRED)  # 2, 9, 8, 13


# The horizontal band strip: rows30-33 hold colour bands stacked HORIZONTALLY
# (gray slot cols8-11 | red bar cols12-17 | darkred base cols18-21).
STRIP_ROWS = (30, 33)


def _region_has(grid, region, color):
    """True if any non-green cell of the region currently shows `color`."""
    r0, r1, c0, c1 = region
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r][c] == color:
                return True
    return False


def _checker_fill(grid, region, r, c, color=BLUE):
    """Static fill of a toggleable BLOB region cell (BLUE or WHITE family). The
    region is either SOLID (all `color`) or CHECKER ((r+c)%2==1 -> color else bg
    4). A checker region always leaves bg(4) cells visible, so any non-green 4 in
    the region means checker."""
    if _region_has(grid, region, LEFT_BG):
        return color if (r + c) % 2 == 1 else LEFT_BG
    return color


def _span_cell(units, idx, n):
    """Colour of cell `idx` of an n-wide navy/teal transfer span holding `units`
    material units (0..16): VOID when empty, TEAL when full, else navy growing
    inward from both ends (navy count == units) over a void middle. The single
    shared rule used by every span renderer/reader (L4 strip+reservoir, L5)."""
    if units <= 0:
        return VOID
    if units >= 16:
        return TEAL
    nend = units // 2
    return NAVY if (idx < nend or idx >= n - nend) else VOID


def _bg_fill(grid, r, c):
    """Static background colour under a vacated green footprint cell.

    Both levels share the same shape: a fixed composite background whose only
    dynamic regions are the toggle pieces (red/pink bar H<->V, blue solid<->
    checker), read directly from each region's still-visible (non-green) cells.
    The geometry differs per level, so dispatch on the detected level."""
    fn = _BG_FILL_BY_LEVEL.get(_detect_level(grid), _bg_fill_generic)
    return fn(grid, r, c)   # unknown level: general sampler


# Cells that are NEVER floor: the two panel backgrounds, the outer border and
# the HUD strip. Everything else (any solid colour region) is walkable floor --
# the general world-model rule confirmed across L1-L3.
_WALL_COLORS = (LEFT_BG, RIGHT_BG, 3, 0)


def _bg_fill_generic(grid, r, c):
    """PROVISIONAL static-background reconstruction for an unmodelled level (L4).

    No L4 transitions exist in the buffer yet, so the per-region geometry is
    unknown. As a safe default the vacated cell reverts to the nearest static
    floor colour around it (the green token slides over a static composite of
    solid regions). This is a placeholder until L4 transitions are observed and
    a derived/region-aware fill replaces it; it is never exercised by the
    verifier (no L4 steps) and only used by the planner's own simulation."""
    h, w = len(grid), len(grid[0])
    for rad in range(1, 6):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = grid[nr][nc]
                    if v != GREEN and v not in _WALL_COLORS and v != YELLOW:
                        return v
    return LEFT_BG


def _l4_bridge_k(grid):
    """Number of GRAY-block bridge clicks applied so far, read live from the
    TEAL reservoir bar (rows40-41 cols16-31). Each click widens the teal central
    void notch by 4 cells (2 per side), so void-count/4 == clicks. 0 == teal
    still solid (no bridge built yet)."""
    vc = sum(1 for c in range(16, 32) if grid[40][c] == VOID)
    return vc // 4


def _l4_barcolor(grid, rows):
    """Current colour (NAVY before docking / TEAL after) of a horizontal slide
    bar occupying the given row pair, read from its still-visible cells. Used to
    revert a cell vacated by green that sits on the bar."""
    for rr in rows:
        for cc in range(4, 24):
            if grid[rr][cc] == TEAL:
                return TEAL
            if grid[rr][cc] == NAVY:
                return NAVY
    return LEFT_BG


def _bg_fill_l4(grid, r, c):
    """L4 static background under a vacated green footprint cell (r, c).

    The only dynamic floor near green's reachable area is the VOID strip
    (rows20-21 cols8-23): the GRAY-block bridge fills navy(1) inward from both
    ends, the rest staying void(5). The navy extent is read live from the teal
    reservoir (`_l4_bridge_k`). Everything else green walks over is static gray
    framing; fall back to the nearest-floor sampler outside the mapped regions."""
    # Void bridge strip: navy at the built ends, void in the unbridged middle.
    # Once the last gap closes the whole span SOLIDIFIES to teal (the finished
    # bridge colour, walkable floor), detected by any teal still in the strip.
    if 20 <= r <= 21 and 8 <= c <= 23:
        if any(grid[20][cc] == TEAL or grid[21][cc] == TEAL
               for cc in range(8, 24)):
            return TEAL
        k = _l4_bridge_k(grid)
        if c <= 7 + 2 * k or c >= 24 - 2 * k:
            return NAVY
        return VOID
    # TEAL reservoir transfer span (rows40-41 cols16-31): green walks over it
    # when full (solid teal floor). Vacated cells revert to the reservoir's live
    # fill (void / navy-ends / teal), read from the visible span (green counts).
    if 40 <= r <= 41 and 16 <= c <= 31:
        units = _l4_span_units(grid, _L4_RES_ROWS[0], _L4_RES_COLS)
        return _span_cell(units, c - _L4_RES_COLS[0], len(_L4_RES_COLS))
    # Pink/darkpink TELEPORT portal tokens (read live phase).
    if 24 <= r <= 25 and 4 <= c <= 5:
        return _mag_t1_val(r, c, _l4_mag_phase(grid))
    if 32 <= r <= 33 and 18 <= c <= 19:
        return _mag_t2_val(r, c, _l4_mag_phase(grid))
    # Horizontal NAVY/TEAL slide bars: bar A rows28-29, bar B rows38-39, in the
    # left workspace (cols<=23; gray bar A at cols24-27 is handled below). The
    # bar's current colour (navy before docking, teal after) is read live from
    # its still-visible cells.
    if (28 <= r <= 29 or 38 <= r <= 39) and c <= 23:
        return _l4_barcolor(grid, (28, 29) if r <= 29 else (38, 39))
    # Blue blob regions (solid <-> checker), read live. Checked AFTER the navy
    # bars: bar B (rows38-39) slides over blue region 2's top rows, so a vacated
    # bar-row cell reverts to navy (handled above), not the blue blob beneath.
    for reg in BLUE4_REGIONS:
        r0, r1, c0, c1 = reg
        if r0 <= r <= r1 and c0 <= c <= c1:
            return _checker_fill(grid, reg, r, c)
    # Green's L4 home frame (gray) rows34-37 cols30-33 (this L4 variant).
    if 34 <= r <= 37 and 30 <= c <= 33:
        return FRAME
    # Player home frame (gray) rows18-21 cols4-7.
    if 18 <= r <= 21 and 4 <= c <= 7:
        return FRAME
    # Right gray bar A rows18-29 cols24-27.
    if 18 <= r <= 29 and 24 <= c <= 27:
        return FRAME
    # Pink-token-1 gray frame rows24-27 cols4-9 (token cells handled elsewhere).
    if 24 <= r <= 27 and 4 <= c <= 9:
        return FRAME
    # Gray bar B rows30-37 cols4-9.
    if 30 <= r <= 37 and 4 <= c <= 9:
        return FRAME
    # Pink-token-2 gray frame rows30-33 cols16-19.
    if 30 <= r <= 33 and 16 <= c <= 19:
        return FRAME
    # Yellow target gray frame rows42-45 cols28-31.
    if 42 <= r <= 45 and 28 <= c <= 31:
        return FRAME
    return _bg_fill_generic(grid, r, c)


def _bg_fill_l5(grid, r, c):
    """L5 static background under a vacated green footprint cell (r, c).

    Green's observed reachable area (steps 380-397): a gray vertical block
    (rows30-33 cols26-29), the BLUE blob region below it (rows34-37 cols26-29,
    BLUE4 geometry, solid<->checker live), and its gray home frame (rows34-37
    cols30-33). Also map the single transfer span (rows44-45 cols14-29) so a
    vacated span cell reverts to its live fill. Everything else falls back to
    the nearest-floor sampler."""
    # Blue blob regions (solid <-> checker), read live.
    for reg in BLUE4_REGIONS:
        r0, r1, c0, c1 = reg
        if r0 <= r <= r1 and c0 <= c <= c1:
            return _checker_fill(grid, reg, r, c)
    # Purple blob regions (solid <-> checker), read live. Same toggle family as
    # the blue blob; solid purple(15) is walkable floor, so green descends through
    # it and vacated cells must revert to the live purple fill (buffer steps
    # 640-642, 647: green walks DOWN purpleA r46-53 c26-29). Checked before the
    # teal vertical scan so the teal bridge directly above (rows44-45) is not
    # mis-grabbed as the revert colour.
    for reg in PURPLE5_REGIONS:
        r0, r1, c0, c1 = reg
        if r0 <= r <= r1 and c0 <= c <= c1:
            return _checker_fill(grid, reg, r, c, color=PURPLE)
    # Gray vertical block above the blue region.
    if 30 <= r <= 33 and 26 <= c <= 29:
        return FRAME
    # Lower-left gray platform (cols10-13 rows34-45) carrying the L5_TOK2 portal
    # token (rows34-35 cols10-11, live phase) and the blue blob (rows38-41,
    # handled by the BLUE4 check above). Buffer steps 554/555: green descends this
    # column; vacated cells revert to the portal token, then plain gray.
    if 34 <= r <= 45 and 10 <= c <= 13:
        if 34 <= r <= 35 and 10 <= c <= 11:
            return _mag_t2_val(r, c, _portal_phase(grid, L5_TOK1, L5_TOK2))
        return FRAME
    # Green's gray home frame.
    if 34 <= r <= 37 and 30 <= c <= 33:
        return FRAME
    # Transfer span (rows44-45 cols14-29): live fill (void / navy-ends / teal).
    if 44 <= r <= 45 and 14 <= c <= 29:
        units = _l4_span_units(grid, _L5_SPAN_ROWS[0], _L5_SPAN_COLS)
        return _span_cell(units, c - _L5_SPAN_COLS[0], len(_L5_SPAN_COLS))
    # Static gray platform block (rows20-23 cols26-29): green walks over it; it
    # sits adjacent to, but is NEVER occupied by, the movable teal bar (no teal
    # ever appears here in the buffer). Buffer steps 542/543: vacated cells
    # revert to gray, not the neighbouring teal bar -- so map it explicitly
    # before the teal-scan / generic sampler can mis-grab the adjacent teal.
    if 20 <= r <= 23 and 26 <= c <= 29:
        return FRAME
    # Upper portal-token platform (rows6-9 cols22-25): a gray pillar carrying
    # the P1 teleport token at rows6-7 cols24-25; rows8-9 are plain gray floor.
    # Buffer step 551: green vacates rows8-9 c24-25 -> gray, not the teal bar
    # directly below (rows10+). Map gray here, portal value on the token cells.
    if 6 <= r <= 9 and 22 <= c <= 25:
        if 6 <= r <= 7 and 24 <= c <= 25:
            return _mag_t1_val(r, c, _portal_phase(grid, L5_TOK1, L5_TOK2))
        return FRAME
    # PURPLE token's gray frame (rows18-21 cols10-13): the consumable purple
    # token sits in a gray(2) frame whose lower rows (20-21) are plain floor green
    # starts on. Vacated cells revert to gray, NOT the teal bar directly below
    # (rows22+). Buffer step 602: green steps up onto the token; the cells it
    # leaves (rows20-21) revert to gray frame. Map before the teal column-scan.
    if 18 <= r <= 21 and 10 <= c <= 13:
        return FRAME
    # Movable NAVY bar (rows54-55): green rides it (buffer steps 647-651). The
    # bar slides via the yellow anvil (period-6 dock/wrap, NAVY<->TEAL on dock).
    # A vacated cell embedded in the bar reverts to the bar colour; identify it
    # by scanning horizontally past green's own footprint for a NAVY/TEAL cell
    # in these rows (the only such bar lives here; the transfer span is rows44-45).
    if r in _L5_NAVY_BAR_ROWS:
        for dc in (-1, 1):
            nc = c + dc
            while 0 <= nc < len(grid[0]) and grid[r][nc] == GREEN:
                nc += dc
            if 0 <= nc < len(grid[0]) and grid[r][nc] in (NAVY, TEAL):
                return grid[r][nc]
    # Movable teal bridge bar (vertical): a vacated green cell embedded in the
    # bar reverts to teal (buffer step 493: green slid left along the bar). The
    # bar is the only teal in rows<44, so a teal cell vertically adjacent (past
    # green's own footprint) identifies it.
    for dr in (-1, 1):
        nr = r + dr
        while 0 <= nr < len(grid) and grid[nr][c] == GREEN:
            nr += dr
        if 0 <= nr < len(grid) and grid[nr][c] == TEAL:
            return TEAL
    return _bg_fill_generic(grid, r, c)


def _bg_fill_l6(grid, r, c):
    """L6 static background under a vacated green footprint cell (r, c).

    PROVISIONAL: only 7 L6 steps are observed (buffer 659-665), all green slides
    through the central gray(2) playfield box (rows44-53 cols18-29) that carries
    a pink/darkpink portal token at its top-left (rows48-49 cols18-19). A vacated
    box cell reverts to gray; the nearest-floor sampler must NOT be used there
    because it mis-grabs the adjacent portal colour. Everything outside the
    observed box defers to the sampler. This is modelling debt -- the full L6
    region map (other portals, red/navy/teal bars) is not yet derived."""
    # Gray room ABOVE the left tower (rows 16-23 cols 4-11): a plain gray(2)
    # playfield holding the multi-colour consumable token (buffer steps 709-716).
    # Vacated footprint cells revert to gray; the nearest-floor sampler must NOT
    # be used (it mis-grabs the token's red/pink/yellow cells or the teal tower
    # just below at row 24).
    if 16 <= r <= 23 and 4 <= c <= 11:
        return FRAME
    # Left ANVIL TOWER (cols 8-11): a teal(12) vertical tower with an embedded
    # red-framed token box (rows 32-35) and a gray(2) base (rows 44-53). Green
    # climbs cols 10-11 up through it (buffer steps 696-708); vacated cells must
    # revert to this static structure, NOT the nearest-floor sampler (which
    # mis-grabs the navy conveyor below or the red box). The red box is a red(8)
    # outline around a teal(12) 2x2 centre.
    if 8 <= c <= 11:
        if 24 <= r <= 43:
            if 32 <= r <= 35:
                if r in (32, 35) or c in (8, 11):
                    return RED
                return TEAL
            return TEAL
        if 44 <= r <= 53:
            return FRAME
    # Central gray playfield box: solid gray except the portal token corner.
    if 44 <= r <= 53 and 18 <= c <= 29:
        # The pink/darkpink teleport portal token sits at the box top-left
        # (rows48-49 cols18-19); a vacated cell there reverts to its live phase.
        t0, t1, t2, t3 = L6_TOK1
        if t0 <= r <= t1 and t2 <= c <= t3:
            paint = _l6_tok1_paint(grid)
            if paint is not None:
                pd, pl = paint
                return pd if (r + c) % 2 == 0 else pl
            return _mag_t1_val(r, c, _portal_phase(grid, L6_TOK1, L6_TOK2))
        return FRAME
    # greenB ROOM (right-side portal room, rows48-53 cols32-37): a plain gray(2)
    # playfield whose bottom-left holds the default portal TOK2 (rows52-53
    # cols32-33). Green teleports in here via a default-portal magenta press and
    # walks UP toward the consumable red block; a vacated TOK2 cell reverts to
    # its live portal phase, the rest of the room to gray (buffer steps 767/768).
    if 48 <= r <= 53 and 32 <= c <= 37:
        t0, t1, t2, t3 = L6_TOK2
        if t0 <= r <= t1 and t2 <= c <= t3:
            return DARKPINK if (r + c) % 2 == _l6_global_phase(grid) else PINK
        return FRAME
    # Conveyor lanes (rows 54-59): green rides a navy/teal bar; a vacated cell
    # reverts to that bar's colour (buffer steps 683/684/686). Scan the row
    # outward for the nearest bar cell (skipping the footprint's other green
    # cells); green covers <=2 cells of a >=6-wide bar so an adjacent bar cell
    # of the ridden bar is always found within the bar width.
    if 54 <= r <= 59:
        w = len(grid[0])
        for d in range(1, 8):
            for cc in (c - d, c + d):
                if 0 <= cc < w and grid[r][cc] in (NAVY, TEAL):
                    return grid[r][cc]
    return _bg_fill_generic(grid, r, c)


def _bg_fill_l3(grid, r, c):
    """L3 static background under a vacated green footprint cell (r, c).

    L3 reuses the SAME region families as L1/L2: a gray-framed player home, four
    toggleable blue regions (solid<->checker), the RED bar (RED2 geometry: V-top
    arm r16-23 c20-23, darkred pivot r24-27 c20-23, V-bot arm r28-35 c20-23, H
    arms r24-27 c12-19 & c24-31, read live via `_red2_in_v`), a gray ledge, and
    the yellow target frame. Only the blue regions and the red bar are dynamic."""
    # Player home frame (gray board behind the green token's start).
    if 16 <= r <= 19 and 28 <= c <= 31:
        return FRAME
    # Magenta portal token 1 + its gray frame (rows16-19 cols4-7).
    if 16 <= r <= 19 and 4 <= c <= 7:
        if 16 <= r <= 17 and 6 <= c <= 7:
            return _mag_t1_val(r, c, _mag_phase(grid))
        return FRAME
    # Magenta portal token 2 + its gray frame (rows44-47 cols12-15).
    if 44 <= r <= 47 and 12 <= c <= 15:
        if 46 <= r <= 47 and 12 <= c <= 13:
            return _mag_t2_val(r, c, _mag_phase(grid))
        return FRAME
    # Consumed white-token frame (gray board once the white anvil is consumed).
    if 40 <= r <= 43 and 16 <= c <= 19:
        return FRAME
    # Four toggleable blue regions (solid <-> checker), read live.
    for reg in BLUE3_REGIONS:
        r0, r1, c0, c1 = reg
        if r0 <= r <= r1 and c0 <= c <= c1:
            return _checker_fill(grid, reg, r, c)
    # White blob regions (solid <-> checker), read live.
    for reg in WHITE3_REGIONS:
        r0, r1, c0, c1 = reg
        if r0 <= r <= r1 and c0 <= c <= c1:
            return _checker_fill(grid, reg, r, c, WHITE)
    # Red bar V-top arm (present only when red=V).
    if 16 <= r <= 23 and 20 <= c <= 23:
        return RED if _red2_in_v(grid) else LEFT_BG
    # Red bar row: gray ledge | (blue block handled above) | H arm | darkred
    # pivot | H arm.
    if 24 <= r <= 27:
        if 4 <= c <= 7:
            return FRAME
        if 12 <= c <= 19:
            return RED if not _red2_in_v(grid) else LEFT_BG
        if 20 <= c <= 23:
            return DARKRED
        if 24 <= c <= 31:
            return RED if not _red2_in_v(grid) else LEFT_BG
    # Red bar V-bot arm (present only when red=V).
    if 28 <= r <= 35 and 20 <= c <= 23:
        return RED if _red2_in_v(grid) else LEFT_BG
    # Yellow target token + its gray frame.
    if 36 <= r <= 39 and 30 <= c <= 33:
        if 38 <= r <= 39 and 32 <= c <= 33:
            return YELLOW
        return FRAME
    return LEFT_BG


def _bg_fill_l2(grid, r, c):
    """L2 static background. Regions (from l2 entry layout): yellow token frame,
    the red/darkred bar row, the gray + blue-block band the player walks, the
    blue checker column, and the pink bar. Only the blue regions are dynamic."""
    # Yellow target token + its gray frame.
    if 12 <= r <= 15 and 20 <= c <= 23:
        if 12 <= r <= 13 and 22 <= c <= 23:
            return YELLOW
        return FRAME
    # Red bar (toggleable after the red token is consumed). Read its current
    # H/V state live; before unlock it is always H, so this is consistent.
    # V up-arm (above the pivot), present only when red=V.
    if 16 <= r <= 23 and 20 <= c <= 23:
        return RED if _red2_in_v(grid) else LEFT_BG
    # Red bar row: gray | red H-arm | darkred pivot | red H-arm | gray.
    if 24 <= r <= 27:
        if 8 <= c <= 11:
            return FRAME
        if 12 <= c <= 19:
            return RED if not _red2_in_v(grid) else LEFT_BG
        if 20 <= c <= 23:
            return DARKRED
        if 24 <= c <= 31:
            return RED if not _red2_in_v(grid) else LEFT_BG
        if 32 <= c <= 35:
            return FRAME
    # V down-arm (below the pivot), present only when red=V.
    if 28 <= r <= 35 and 20 <= c <= 23:
        return RED if _red2_in_v(grid) else LEFT_BG
    # Consumed red-token frame (gray board once the token is gone).
    if 52 <= r <= 55 and 16 <= c <= 19:
        return FRAME
    # Gray ledge + solid/checker blue block (the band the player climbs).
    if 28 <= r <= 31:
        if 4 <= c <= 7:
            return FRAME
        if 8 <= c <= 11:
            return _checker_fill(grid, BLUE2_BLOCK, r, c)
    # Blue checker column.
    if 32 <= r <= 39 and 4 <= c <= 7:
        return _checker_fill(grid, BLUE2_CHECKER, r, c)
    # Pink-V top arm (above the pivot), present only when pink=V.
    if 32 <= r <= 39 and 16 <= c <= 19:
        return PINK if _pink_in_v(grid) else LEFT_BG
    # Pink bar row: gray ledge | pink H-left arm | darkpink pivot | pink H-right.
    if 40 <= r <= 43:
        if 4 <= c <= 7:
            return FRAME
        if 8 <= c <= 15:
            return PINK if not _pink_in_v(grid) else LEFT_BG
        if 16 <= c <= 19:
            return DARKPINK
        if 20 <= c <= 27:
            return PINK if not _pink_in_v(grid) else LEFT_BG
    # Pink-V bottom arm (below the pivot), present only when pink=V.
    if 44 <= r <= 51 and 16 <= c <= 19:
        return PINK if _pink_in_v(grid) else LEFT_BG
    return LEFT_BG


def _bg_fill_l1(grid, r, c):
    """L1 static background under a vacated green footprint cell (r, c).

    The level-1 layout is fixed; only the red(H/V) and blue(solid/checker)
    toggles change a region's fill, and those states are read directly from the
    region's still-visible (non-green) cells. This replaces the old fragile
    nearest-floor sampling that mis-picked the yellow-frame gray at corridor
    edges (cols 21 vs the yellow frame at col 22)."""
    # Green home frame (gray board).
    if 38 <= r <= 43 and 8 <= c <= 13:
        return FRAME
    # Blue blob bridge B (solid/checker).
    if 34 <= r <= 37 and 8 <= c <= 11:
        return _checker_fill(grid, BLUE_B, r, c)
    # Gray slot pocket.
    if 30 <= r <= 33 and 8 <= c <= 11:
        return FRAME
    # Red bar (present only when red=H).
    if 30 <= r <= 33 and 12 <= c <= 17:
        return RED if _region_has(grid, RED_H, RED) else LEFT_BG
    # Darkred base.
    if 30 <= r <= 33 and 18 <= c <= 21:
        return DARKRED
    # Red V-segment (present only when red=V).
    if 24 <= r <= 29 and 18 <= c <= 21:
        return RED if _region_has(grid, RED_V, RED) else LEFT_BG
    # Blue-A region (solid/checker).
    if 20 <= r <= 23 and 18 <= c <= 21:
        return _checker_fill(grid, BLUE_A, r, c)
    # Yellow target frame (gray board with a yellow 2x2 token at its centre).
    if 18 <= r <= 23 and 22 <= c <= 27:
        if 20 <= r <= 21 and 24 <= c <= 25:
            return YELLOW
        return FRAME
    return LEFT_BG


def _move_delta(action_id):
    if action_id == A_UP:
        return -2, 0
    if action_id == A_DOWN:
        return 2, 0
    if action_id == A_LEFT:
        return 0, -2
    if action_id == A_RIGHT:
        return 0, 2
    return None


def _green_dest(grid, action_id):
    """Destination bbox (nt,nl,nb,nr) of the green token after a move, or None
    (no player / non-move action / off-canvas)."""
    d = _move_delta(action_id)
    pbb = _find_player(grid)
    if d is None or pbb is None:
        return None
    dr, dc = d
    pt, pl, pb, pr = pbb
    nt, nl = pt + dr, pl + dc
    nb, nr = pt + (pb - pt) + dr, pl + (pr - pl) + dc
    h, w = len(grid), len(grid[0])
    if nt < 0 or nl < 0 or nb >= h or nr >= w:
        return None
    return (nt, nl, nb, nr)


def _components(grid, color):
    """Connected components (4-neighbour) of `color`; each as a bbox+size dict."""
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0] or grid[r0][c0] != color:
                continue
            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] \
                            and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            comps.append({"bbox": (min(rs), min(cs), max(rs), max(cs)),
                          "size": len(cells)})
    return comps


def _find_yellow(grid):
    """Bbox (top,left,bottom,right) of the yellow TARGET token, or None.

    The delivery target is the small framed yellow 2x2 on the LEFT workspace
    board. Later levels (L4) also draw a large yellow ANVIL control in the RIGHT
    panel (cols>=34); taking the bbox over all yellow cells would merge the two
    into one giant box and break the win predicate. So pick the yellow component
    that lies in the left workspace; among those, the smallest (the token)."""
    comps = _components(grid, YELLOW)
    if not comps:
        return None
    left = [k for k in comps if k["bbox"][3] < RIGHT_X]
    pool = left if left else comps
    pool.sort(key=lambda k: k["size"])
    return pool[0]["bbox"]


def _is_winning_move(grid, action_id):
    """True iff the move would carry green ONTO the yellow target token (the
    delivery that completes the level)."""
    dest = _green_dest(grid, action_id)
    y = _find_yellow(grid)
    if dest is None or y is None:
        return False
    nt, nl, nb, nr = dest
    yt, yl, yb, yr = y
    overlap = not (nr < yl or nl > yr or nb < yt or nt > yb)
    return overlap


def _move_player(grid, action_id):
    """Slide the green token one cell (2 px). Green is a free 2x2 sprite over a
    static composite background: a move is allowed iff every destination cell is
    a floor colour; vacated cells revert to the static background."""
    out = [list(row) for row in grid]
    pbb = _find_player(grid)
    dest = _green_dest(grid, action_id)
    if pbb is None or dest is None:
        return out
    pt, pl, pb, pr = pbb
    nt, nl, nb, nr = dest

    # Destination must be entirely floor (token moves 2px so destination cells do
    # not overlap the old footprint; read straight from the input grid). L2 adds
    # the pink bar (pink=7 arms + darkpink=6 pivot) as walkable floor.
    # Unknown level (not in the table) -> floor=None uses the GENERAL rule below.
    lvl = _detect_level(grid)
    floor = _FLOOR_BY_LEVEL.get(lvl)
    for r in range(nt, nb + 1):
        for c in range(nl, nr + 1):
            v = grid[r][c]
            # The L6 HOME portal token TOK1 is walkable in any paint colour (it
            # carries non-floor teal/yellow/lblue checker cells once painted, yet
            # green steps onto it -- buffer step 758 ACTION1 onto a teal/yellow
            # painted TOK1). Treat its whole 2x2 as floor.
            if lvl == 6 and (L6_TOK1[0] <= r <= L6_TOK1[1]
                             and L6_TOK1[2] <= c <= L6_TOK1[3]):
                continue
            if floor is None:
                # General rule: any solid colour is floor; only panel bg, border,
                # HUD and the green token itself block (yellow target is steppable).
                if v in _WALL_COLORS or v == GREEN:
                    return out
            elif v not in floor:
                return out  # blocked: unchanged

    # Reconstruct the static background under the token's current footprint.
    for r in range(pt, pb + 1):
        for c in range(pl, pr + 1):
            out[r][c] = _bg_fill(grid, r, c)

    for r in range(nt, nb + 1):
        for c in range(nl, nr + 1):
            out[r][c] = GREEN
    return out


# --- Preview-click piece mechanics ------------------------------------------

# Red workspace piece, two toggle positions (inclusive bbox: r0,r1,c0,c1).
RED_H = (30, 33, 12, 17)   # horizontal bar slot
RED_V = (24, 29, 18, 21)   # vertical, stacked above the darkred base
# Two blue workspace regions (each toggles solid <-> checker).
BLUE_A = (20, 23, 18, 21)
BLUE_B = (34, 37, 8, 11)


def _region_count(grid, region, color):
    r0, r1, c0, c1 = region
    return sum(1 for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
              if grid[r][c] == color)


def _region_full(grid, region, color):
    """True iff every cell of `region` is `color` (a blob region is fully solid)."""
    r0, r1, c0, c1 = region
    return _region_count(grid, region, color) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _green_in_regions(grid, regions):
    """True if the green player token overlaps any of the given regions. A
    toggle that would reshape a piece the player is standing on is refused by
    the env (you cannot pull the floor out from under the token), so such a
    click is a pure no-op (observed L2 step 158: blue click while green sits on
    the blue block changed nothing but the HUD)."""
    for r0, r1, c0, c1 in regions:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r][c] == GREEN:
                    return True
    return False


def _fill_region(out, region, color):
    r0, r1, c0, c1 = region
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r][c] = color


def _right_has_white(grid):
    """True if either preview box still carries its one-time white(0) outline."""
    for r in range(14, 40):
        for c in range(40, 57):
            if grid[r][c] == 0:
                return True
    return False


def _redraw_plain_previews(out):
    """Remove one-time white outlines and redraw both previews as plain solid
    anvils (the settled after-form seen on the first preview click)."""
    # Clear both preview boxes to right-panel bg.
    for r in range(16, 23):
        for c in range(41, 56):
            out[r][c] = RIGHT_BG
    for r in range(33, 40):
        for c in range(41, 56):
            out[r][c] = RIGHT_BG
    # Plain red anvil: top block rows17-19 cols45-51, base rows20-21 cols42-54.
    for r in range(17, 20):
        for c in range(45, 52):
            out[r][c] = RED
    for r in range(20, 22):
        for c in range(42, 55):
            out[r][c] = RED
    # Plain blue anvil: top rows34-36 cols45-51, base rows37-38 cols42-54.
    for r in range(34, 37):
        for c in range(45, 52):
            out[r][c] = BLUE
    for r in range(37, 39):
        for c in range(42, 55):
            out[r][c] = BLUE


# --- Shared toggle families --------------------------------------------------
# Two toggle FAMILIES recur across all levels; both are pure functions of the
# frame, both refuse to fire while green stands on an affected region, and both
# strip the one-time white preview outline on the first click.
#   BAR:  a coloured bar flips between Horizontal arm regions and Vertical arm
#         regions around a fixed dark-shade pivot (red L1, red/pink L2).
#   BLOB: a set of blue regions each swap SOLID <-> CHECKER (always paired in
#         opposite phase, so toggling all of them swaps which is solid).

def _bar_is_v(grid, color, h_regions, v_regions):
    """True when the bar currently shows its Vertical arms (more `color` cells
    in the V regions than the H regions)."""
    v = sum(_region_count(grid, r, color) for r in v_regions)
    h = sum(_region_count(grid, r, color) for r in h_regions)
    return v >= h


def _toggle_bar(grid, color, h_regions, v_regions, redraw=False):
    """Flip a bar H<->V. No-op while green overlaps any of its arm regions."""
    if _green_in_regions(grid, tuple(h_regions) + tuple(v_regions)):
        return [list(row) for row in grid]
    out = [list(row) for row in grid]
    if redraw and _right_has_white(out):
        _redraw_plain_previews(out)
    show_h = _bar_is_v(out, color, h_regions, v_regions)  # currently V -> go H
    on, off = (h_regions, v_regions) if show_h else (v_regions, h_regions)
    for reg in off:
        _fill_region_safe(out, reg, LEFT_BG)
    for reg in on:
        _fill_region_safe(out, reg, color)
    return out


def _toggle_blue_region(out, region, color=BLUE):
    r0, r1, c0, c1 = region
    solid = all(out[r][c] == color
                for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if solid:
                out[r][c] = color if (r + c) % 2 == 1 else LEFT_BG
            else:
                out[r][c] = color


def _toggle_blue(grid, regions, redraw=False, color=BLUE):
    """Swap a set of BLOB regions solid<->checker (BLUE or WHITE family). No-op
    while green overlaps."""
    if _green_in_regions(grid, regions):
        return [list(row) for row in grid]
    out = [list(row) for row in grid]
    if redraw and _right_has_white(out):
        _redraw_plain_previews(out)
    for reg in regions:
        _toggle_blue_region(out, reg, color)
    return out


def _click_red(grid):
    return _toggle_bar(grid, RED, (RED_H,), (RED_V,), redraw=True)


def _click_blue(grid):
    return _toggle_blue(grid, (BLUE_A, BLUE_B), redraw=True)


# --- Level 2 piece mechanics -------------------------------------------------
# L2 reuses the SAME toggle FAMILIES as L1 (a colored BAR toggling H<->V, and a
# blue BLOB swapping solid<->checker between two regions); only the geometry and
# the trigger-button colour differ. The magenta(6) right-panel anvil toggles the
# PINK(7) bar; the blue(9) anvil swaps the two blue regions' phase.
PINK = 7
DARKPINK = 6          # pink-bar pivot colour AND the magenta button colour
# Pink bar Horizontal: two arms either side of the darkpink pivot (cols16-19).
PINK_H_LEFT = (40, 43, 8, 15)
PINK_H_RIGHT = (40, 43, 20, 27)
# Pink bar Vertical: arms above AND below the fixed darkpink pivot (rows40-43).
PINK_V_TOP = (32, 39, 16, 19)
PINK_V_BOT = (44, 51, 16, 19)
# Two blue regions (L2 layout): a solid block and a checker blob that swap phase.
BLUE2_BLOCK = (28, 31, 8, 11)
BLUE2_CHECKER = (32, 39, 4, 7)

# L2 floor adds the pink bar: pink(7) H/V arms and the darkpink(6) pivot are
# walkable (buffer steps 170-174: green slides right along the pink-H bar and
# onto the darkpink pivot).
_FLOOR_L2 = (FRAME, BLUE, RED, DARKRED, PINK, DARKPINK)

# L2 RED-TOKEN UNLOCK (buffer steps 181-182). A small red(8) anvil-shaped token
# sits inside a gray(2) frame at rows52-55 cols16-19. When green's move DELIVERS
# onto it, the token is CONSUMED (its red cells revert to gray frame) and a RED
# ANVIL button SPAWNS in the right panel. From then on the L2 RED BAR (rows24-27)
# becomes a clickable H<->V toggle (same bar family as L1 red / L2 pink), its V
# arms bridging cols20-23 up (rows16-23) to the isolated yellow token frame.
RED_TOKEN = (52, 55, 16, 19)        # the consumable red token's gray frame
# Right-panel red anvil that appears after the token is consumed.
RED_ANVIL_STEM = (29, 31, 49, 55)   # 7-wide stem block
RED_ANVIL_BASE = (32, 33, 46, 58)   # 13-wide base
# L2 red bar toggle regions (post-unlock). Pivot = darkred(13) rows24-27 c20-23.
RED2_H_LEFT = (24, 27, 12, 19)
RED2_H_RIGHT = (24, 27, 24, 31)
RED2_V_TOP = (16, 23, 20, 23)
RED2_V_BOT = (28, 35, 20, 23)


def _red2_in_v(grid):
    return _bar_is_v(grid, RED, (RED2_H_LEFT, RED2_H_RIGHT),
                     (RED2_V_TOP, RED2_V_BOT))


# --- Shared consumable-token mechanic ---------------------------------------
# One rule recurs (L2 red, L3 white): a coloured anvil token sits in a gray(2)
# frame; DELIVERING green onto it CONSUMES it (its coloured cells revert to the
# gray frame) and SPAWNS a same-coloured anvil button in the right panel that
# unlocks a toggle. Per-level data: (token_box, colour, (anvil_regions...)).
def _has_token(grid, box, color):
    r0, r1, c0, c1 = box
    return any(grid[r][c] == color
               for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))


def _move_lands_on_token(grid, action_id, box, color):
    """True iff this move carries green onto a still-present coloured token cell."""
    if not _has_token(grid, box, color):
        return False
    dest = _green_dest(grid, action_id)
    if dest is None:
        return False
    nt, nl, nb, nr = dest
    r0, r1, c0, c1 = box
    for r in range(nt, nb + 1):
        for c in range(nl, nr + 1):
            if r0 <= r <= r1 and c0 <= c <= c1 and grid[r][c] == color:
                return True
    return False


def _consume_token(grid, action_id, box, color, anvils):
    """Slide green onto the token, revert its remaining coloured cells to the
    gray frame, and spawn the coloured anvil button in the right panel."""
    out = _move_player(grid, action_id)
    r0, r1, c0, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r][c] == color:
                out[r][c] = FRAME
    for reg in anvils:
        _fill_region(out, reg, color)
    return out


def _click_red_l2(grid):
    """Toggle the unlocked L2 red bar H<->V around its fixed darkred pivot."""
    return _toggle_bar(grid, RED, (RED2_H_LEFT, RED2_H_RIGHT),
                       (RED2_V_TOP, RED2_V_BOT))


def _fill_region_safe(out, region, color):
    """Fill a region with `color` but never overwrite the player token."""
    r0, r1, c0, c1 = region
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r][c] != GREEN:
                out[r][c] = color


def _detect_level(grid):
    """Return the current level (1..4) from persistent palette signals.

    Each level is identified by a colour that no earlier level uses, checked
    in level order so a later level's superset palette resolves correctly:
      L4 -- introduces NAVY(1) and TEAL(12) pieces (no red/blue/darkred/white).
      L3 -- the only level whose BOARD carries the white(15) checker bridge.
      L2 -- right panel carries the magenta/darkpink(6) anvil button.
      L1 -- red(8) or blue(9) pieces present and none of the above.
      0  -- UNKNOWN: matches no known signature (a novel level). The caller must
            fall back to the GENERAL floor/background rules rather than reusing
            L1's hardcoded geometry (which would silently mispredict).
    Using a unique colour (rather than an exact entry-frame match) keeps the
    detector stable across in-level toggles and green movement."""
    h = len(grid)
    w = len(grid[0])
    # L5/L6 -- both introduce colour 10 (no earlier level uses it) and share an
    # IDENTICAL palette, so colour alone cannot separate them. They are distinct
    # hand-designed boards: L5 is dominated by toggle BLOBS (blue~100 cells over a
    # small gray frame), L6 is a large OPEN gray playfield (gray~416 cells). Split
    # on the gray(FRAME) mass, which is tightly clustered (L5 ~132-152, L6 =416)
    # and unaffected by the green token / observed toggles. Checked BEFORE L4.
    has10 = any(v == 10 for row in grid for v in row)
    if has10:
        gray = sum(1 for row in grid for v in row if v == FRAME)
        return 6 if gray > 250 else 5
    for row in grid:
        for v in row:
            if v == 1 or v == 12:
                return 4
    for r in range(8, min(56, h)):
        for c in range(min(38, w)):
            if grid[r][c] == 15:
                return 3
    for r in range(8, min(56, h)):
        for c in range(40, min(64, w)):
            if grid[r][c] == DARKPINK:
                return 2
    for row in grid:
        for v in row:
            if v == RED or v == BLUE:
                return 1
    return 0


# --- Level 3 piece mechanics -------------------------------------------------
# L3 reuses the SAME toggle FAMILIES again: a RED BAR H<->V (identical geometry
# to the L2 unlocked red bar, RED2_*), a blue BLOB swapping solid<->checker over
# FOUR regions, and a MAGENTA token-pair toggle -- two 2x2 pink/magenta tokens
# whose cells swap 6<->7 in place (the L3 analogue of the bar/blob phase flip).
BLUE3_REGIONS = ((16, 19, 24, 27), (20, 23, 4, 7),
                 (24, 27, 8, 11), (40, 43, 12, 15))
# Two 2x2 magenta TELEPORT PORTAL tokens. Green can walk ONTO either (they are
# floor); clicking the magenta anvil while green stands on one teleports green
# to the OTHER (the vacated portal re-shows its magenta checker pattern). When
# green is on neither, the click swaps both portals' checker phase in place.
# Both tokens are always in OPPOSITE checker phase (one even->6, the other
# even->7), and every magenta press FLIPS the shared phase bit.
MAG_T1 = (16, 17, 6, 7)
MAG_T2 = (46, 47, 12, 13)
MAG3_TOKENS = (MAG_T1, MAG_T2)
WHITE = 15

# L3 consumable WHITE anvil token (buffer step 242): a white(15) anvil sits in a
# gray(2) frame at rows40-43 cols16-19. Delivering green onto its white cells
# consumes it (white -> gray frame) and spawns a WHITE anvil button in the right
# panel (role of that button still unobserved; assumed to unlock the bridge to
# the yellow target).
WHITE_TOKEN = (40, 43, 16, 19)
WHITE_ANVIL_STEM = (43, 45, 48, 54)
WHITE_ANVIL_BASE = (46, 47, 45, 57)

# L3 WHITE BLOB: once the white anvil spawns, clicking it swaps these white(15)
# regions solid<->checker (same BLOB family as blue, just white-coloured). When
# solid the cells become walkable floor and the bridge up cols16-19 is built
# (buffer steps 243 toggle; 246-247 green climbs through the solid white).
WHITE3_REGIONS = ((32, 39, 16, 19), (32, 35, 24, 33))

# L3 walkable floor adds the magenta portal cells (pink7 / darkpink6) and, once
# solid, the white blob (15). In checker phase the blob's bg(4) gaps still block.
_FLOOR_L3 = (FRAME, BLUE, RED, DARKRED, PINK, DARKPINK, WHITE)

# L4 floor (PROVISIONAL -- no L4 transitions observed yet). L4 has none of the
# red/blue/darkred/white pieces; its solid colours are the gray frame, the
# NAVY(1) and TEAL(12) bars, and the pink/darkpink tokens. The general
# world-model rule (green walks any solid colour; walls = panel bg/border/HUD)
# is applied here as the working hypothesis until the buffer shows L4 moves.
NAVY = 1
TEAL = 12
VOID = 5            # L4 black void / wall colour
# Two L4 blue BLOB regions (same solid<->checker family as L1-L3 blue), toggled
# by the blue(9) right-panel anvil (buffer step 379). Solid blue is walkable.
BLUE4_REGIONS = ((34, 37, 26, 29), (38, 41, 10, 13))
_FLOOR_L4 = (FRAME, NAVY, TEAL, PINK, DARKPINK, BLUE)
# L5 floor = L4 floor plus the purple(15) pieces (the consumable purple token
# and, once solid, the purple blob bridge). Checker phase still blocks via bg(4).
_FLOOR_L5 = _FLOOR_L4 + (WHITE,)
# L6 floor (PROVISIONAL, only 7 buffer steps observed): the large open gray
# playfield (FRAME=2) plus the pink/darkpink teleport-portal tokens green steps
# onto (buffer step 665). Red/navy/teal bars are present but their walkability is
# unobserved; kept in the floor set under the general "solid colour is floor"
# rule. Narrow this once more L6 moves are observed.
_FLOOR_L6 = (FRAME, PINK, DARKPINK, RED, NAVY, TEAL, BLUE, WHITE)

# L4 right-panel control anvils (dispatched by the colour under the click):
#   YELLOW(11) anvil  -> slide BOTH navy(1) bars RIGHT by one cell (+2px).
#   DARKPINK(6) anvil -> flip the two pink/darkpink 2x2 tokens' phase (6<->7).
#   GRAY(2) block     -> couple-toggle the void strip ends + the teal bar form.
# (Buffer steps 259/260/261 respectively; only these three L4 steps exist.)

# L4 pink/darkpink token-pair (same family as L3 magenta portals).
L4_TOK1 = (24, 25, 4, 5)
L4_TOK2 = (32, 33, 18, 19)

# L4 gray-block BRIDGE-BUILDER geometry (buffer steps 261, 266). Each click moves
# navy material from the TEAL reservoir bar (rows40-41 cols16-31) into the VOID
# strip (rows20-21 cols8-23): the strip's navy ends grow inward by 2 while the
# teal central void notch grows outward by 2. Handled live in `_click_gray_l4`.


# Per-level dispatch tables (one shared rule, level-keyed data). Unknown levels
# fall back to the GENERAL floor/background rules.
_FLOOR_BY_LEVEL = {1: _FLOOR, 2: _FLOOR_L2, 3: _FLOOR_L3,
                   4: _FLOOR_L4, 5: _FLOOR_L5, 6: _FLOOR_L6}
_BG_FILL_BY_LEVEL = {1: _bg_fill_l1, 2: _bg_fill_l2, 3: _bg_fill_l3,
                     4: _bg_fill_l4, 5: _bg_fill_l5, 6: _bg_fill_l6}


def _green_on_navy_bar_l4(grid):
    """True iff the green token sits ON a navy/teal L4 bar -- i.e. a green cell in
    a bar row (28,29,38,39) whose contiguous horizontal run (through green cells)
    is flanked by a NAVY or TEAL cell. Green merely on the gray dock bar (FRAME)
    or in open bg does NOT count."""
    w = len(grid[0])
    for r in (28, 29, 38, 39):
        for c in range(w):
            if grid[r][c] != GREEN:
                continue
            cl = c
            while cl >= 0 and grid[r][cl] == GREEN:
                cl -= 1
            cr = c
            while cr < w and grid[r][cr] == GREEN:
                cr += 1
            left = grid[r][cl] if cl >= 0 else None
            right = grid[r][cr] if cr < w else None
            if left in (NAVY, TEAL) or right in (NAVY, TEAL):
                return True
    return False


def _click_yellow_l4(grid):
    """YELLOW anvil: the two navy bars cycle RIGHT with PERIOD 6 (buffer steps
    304-311). They move together as one linked object:
      - bar A rows28-29 entry cols8-13, bar B rows38-39 entry cols4-9;
      - each click slides both +2px right (5 slides total);
      - the 5th slide DOCKS them against gray bar A (bar A at cols18-23): both
        bars convert to TEAL(12), the docked/finished colour (buffer step 308);
      - the next click (from the docked teal state) WRAPS both back to their
        entry positions as NAVY(1), clearing the docked teal (buffer step 309).
    Only the two bar row-pairs are affected; navy elsewhere (the void bridge) is
    left untouched. Green parked in a bar's path holds that cell (no overwrite).

    LOCKED UNDER PLAYER -- DOCKED/WRAP ONLY: the no-op lock applies ONLY to the
    WRAP click (from the docked TEAL state). Wrapping teleports both bars from the
    docked position (bar A cols18-23) back to entry (cols8-13); if green stands on
    the docked bar that wrap would strand it off any bar, so the env refuses it
    (buffer step 343: green on docked teal bar A, yellow click changed only HUD).
    A plain SLIDE always fires even with green on a navy bar: the 6-wide bar moves
    +2 under the token, which stays put (buffer steps 357-363: green on navy bar B,
    bars slid, navy re-rendered on both sides of green)."""
    out = [list(row) for row in grid]
    w = len(grid[0])
    BAR_ROWS = (28, 29, 38, 39)
    ENTRY = {28: (8, 13), 29: (8, 13), 38: (4, 9), 39: (4, 9)}
    # DOCKED state: teal present in a bar row -> next click WRAPS both bars back
    # to entry navy (clearing teal). Refused (no-op) if green is on the docked bar.
    docked = any(grid[r][c] == TEAL for r in BAR_ROWS for c in range(4, 28))
    if docked:
        if _green_on_navy_bar_l4(grid):
            return out
        for r in BAR_ROWS:
            for c in range(4, 28):
                if grid[r][c] == TEAL:
                    out[r][c] = LEFT_BG
            e0, e1 = ENTRY[r]
            for c in range(e0, e1 + 1):
                if grid[r][c] != GREEN:
                    out[r][c] = NAVY
        return out
    # SLIDE: each 6-wide navy bar moves +2 right. Recover each bar's full span
    # (green standing on it occludes up to 2 navy cells) then re-render navy on
    # the slid span minus any cell green still covers; green never moves with it.
    spans = {}
    for r in BAR_ROWS:
        navy = [c for c in range(w) if grid[r][c] == NAVY]
        if not navy:
            spans[r] = None
            continue
        L, R = min(navy), max(navy)
        if R - L + 1 < 6:                      # part of bar hidden under green
            if R + 1 < w and grid[r][R + 1] == GREEN:
                R = L + 5                       # green occludes the right end
            elif L - 1 >= 0 and grid[r][L - 1] == GREEN:
                L = R - 5                       # green occludes the left end
            else:
                R = L + 5
        spans[r] = (L, R)
    # The two bars are LINKED; bar A (rows 28-29) leads into the gray dock bar.
    # Decide the shared move from bar A:
    #   - forward cells (R+1,R+2) blocked by GREEN/FRAME -> dock in place (delta 0);
    #   - else slide +2; if the slid right edge then abuts the FRAME -> dock.
    # Docking recolours both bars TEAL; otherwise they stay NAVY.
    LA, RA = spans[28]
    fwd1 = grid[28][RA + 1] if RA + 1 < w else FRAME
    fwd2 = grid[28][RA + 2] if RA + 2 < w else FRAME
    if fwd1 in (GREEN, FRAME) or fwd2 in (GREEN, FRAME):
        delta, dock = 0, True
    else:
        delta = 2
        nr = RA + 2
        dock = (nr + 1 >= w) or grid[28][nr + 1] in (GREEN, FRAME)
    color = TEAL if dock else NAVY
    for r in BAR_ROWS:
        if spans[r] is None:
            continue
        L, R = spans[r]
        for c in range(L, R + 1):
            if grid[r][c] == NAVY:
                out[r][c] = LEFT_BG
        for c in range(L + delta, R + delta + 1):
            if 0 <= c < w and grid[r][c] != GREEN:
                out[r][c] = color
    return out


def _l4_mag_phase(grid):
    return _portal_phase(grid, L4_TOK1, L4_TOK2)


def _click_magenta_l4(grid):
    return _click_portal_pair(grid, L4_TOK1, L4_TOK2)


_L4_STRIP_ROWS = (20, 21)
_L4_STRIP_COLS = list(range(8, 24))
_L4_RES_ROWS = (40, 41)
_L4_RES_COLS = list(range(16, 32))


def _l4_span_units(grid, r0, cols):
    """Material units (0..16) currently held by an L4 transfer span, read live.
    A span holding all material is solid TEAL(16); empty is all VOID(0);
    intermediate is navy-at-both-ends + void-middle (navy count == units).
    GREEN parked on the span masks navy cells, so green cells count toward the
    span's units when teal isn't present."""
    if any(grid[r0][c] == TEAL for c in cols):
        return 16
    navy = sum(1 for c in cols if grid[r0][c] == NAVY)
    green = sum(1 for c in cols if grid[r0][c] == GREEN)
    return navy + green


def _l4_render_span(grid, out, rows, cols, units):
    """Paint a transfer span to `units`: VOID(0) / TEAL(16) / navy-ends with a
    void middle (navy count == units, split evenly across the two ends). GREEN
    cells are preserved (the env reshapes around the parked player)."""
    n = len(cols)
    for idx, c in enumerate(cols):
        v = _span_cell(units, idx, n)
        for r in rows:
            if grid[r][c] != GREEN:
                out[r][c] = v


def _l4_pump_draining(grid):
    """Direction bit for the L4 gray pump, read from the NAVY BAR position.
    The two navy bars cycle RIGHT (yellow-anvil period-6 slide); the pump FILLS
    while the bars are near entry (slid 0-1 steps, bar A leftmost <12) and DRAINS
    once they have been slid >=2 steps (bar A leftmost >=12, i.e. +4px). Endpoints
    override this in the caller (full strip must drain, empty strip must fill).
    This is the only frame-visible correlate of the otherwise hidden pump phase."""
    cols = [c for r in (28, 29) for c in range(4, 32) if grid[r][c] == NAVY]
    return bool(cols) and min(cols) >= 12


def _click_gray_l4(grid):
    """GRAY block anvil: a bidirectional BRIDGE PUMP that transfers 4 material
    units per click between the VOID strip (rows20-21 cols8-23, the bridge green
    walks) and the TEAL reservoir bar (rows40-41 cols16-31). The two spans always
    sum to 16 units (mass conserved): one span solid-teal means the other empty.
    FILL grows the strip's navy ends inward until it is solid teal (bridge built);
    DRAIN reverses it. Direction = `_l4_pump_draining` (navy bar phase), with the
    two endpoints forcing the only physically possible direction.
    GREEN cells are reshaped around (preserved), never overwritten."""
    out = [list(row) for row in grid]
    strip_units = _l4_span_units(grid, _L4_STRIP_ROWS[0], _L4_STRIP_COLS)
    res_units = _l4_span_units(grid, _L4_RES_ROWS[0], _L4_RES_COLS)
    if strip_units >= 16:
        draining = True
    elif strip_units <= 0:
        draining = False
    else:
        draining = _l4_pump_draining(grid)
    if draining:
        if strip_units <= 0:
            return out
        new_strip, new_res = strip_units - 4, res_units + 4
    else:
        if res_units <= 0:
            return out
        new_strip, new_res = strip_units + 4, res_units - 4
    _l4_render_span(grid, out, _L4_STRIP_ROWS, _L4_STRIP_COLS, new_strip)
    _l4_render_span(grid, out, _L4_RES_ROWS, _L4_RES_COLS, new_res)
    return out


# L5 has a SINGLE transfer span on the left board (rows44-45 cols14-29). The
# gray block anvil is a BIDIRECTIONAL pump (same family as the L4 strip/reservoir
# pump, but standalone -- no paired reservoir): it transfers 4 material units per
# click, FILLING the span toward solid teal or DRAINING it toward void. Direction
# is otherwise-hidden state; the only frame-visible correlate is the L5 NAVY BAR
# position (rows54-55), exactly as on L4. Buffer: steps 391/411-414 fill (bar at
# entry, leftmost 14); steps 428/429 drain (bar slid right by the yellow anvil,
# leftmost 16). Endpoints force the only possible direction (full->drain, empty->
# fill).
_L5_SPAN_ROWS = (44, 45)
_L5_SPAN_COLS = list(range(14, 30))
_L5_NAVY_BAR_ROWS = (54, 55)


def _l5_pump_draining(grid):
    """Direction bit for the L5 gray pump, read from the NAVY BAR position
    (rows54-55). The bar slides RIGHT via the yellow anvil; the pump FILLS the
    span while the bar is at entry (leftmost <16) and DRAINS once it has been
    slid right (leftmost >=16). Endpoints override this in the caller.
    DEBT: like the L4 pump, the true pump direction is hidden state; the navy-bar
    position is the only frame-visible correlate and may be a play-order
    coincidence rather than a causal link. Reconfirm if a future buffer fills
    with the bar slid or drains with the bar at entry."""
    cols = [c for c in range(len(grid[0]))
            if grid[_L5_NAVY_BAR_ROWS[0]][c] == NAVY
            or grid[_L5_NAVY_BAR_ROWS[1]][c] == NAVY]
    return bool(cols) and min(cols) >= 16


def _click_gray_l5(grid):
    """GRAY block anvil on L5: bidirectional pump on the single transfer span
    (rows44-45 cols14-29), 4 units per click. Navy grows inward from both ends
    when filling (solid teal when full), void grows from the centre when
    draining. GREEN cells are reshaped around (preserved)."""
    out = [list(row) for row in grid]
    units = _l4_span_units(grid, _L5_SPAN_ROWS[0], _L5_SPAN_COLS)
    if units >= 16:
        draining = True
    elif units <= 0:
        draining = False
    else:
        draining = _l5_pump_draining(grid)
    new_units = units - 4 if draining else units + 4
    new_units = max(0, min(16, new_units))
    _l4_render_span(grid, out, _L5_SPAN_ROWS, _L5_SPAN_COLS, new_units)
    return out


def _click_white_l3(grid):
    """Swap the L3 white blob regions solid<->checker (white BLOB family)."""
    return _toggle_blue(grid, WHITE3_REGIONS, color=WHITE)


def _mag_phase(grid):
    return _portal_phase(grid, MAG_T1, MAG_T2)


def _mag_t1_val(r, c, p):
    return DARKPINK if (r + c) % 2 == p else PINK


def _mag_t2_val(r, c, p):
    return DARKPINK if (r + c) % 2 == (1 - p) else PINK


def _green_on_region(grid, region):
    r0, r1, c0, c1 = region
    return all(grid[r][c] == GREEN
               for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))


def _portal_phase(grid, tok1, tok2):
    """Shared checker-phase bit for a teleport portal pair (L3/L4/L5 magenta):
    token1 cell(r,c) = DARKPINK iff (r+c)%2==p, token2 is the OPPOSITE phase.
    Read from whichever token is currently visible (not masked by green)."""
    for tok, flip in ((tok1, 0), (tok2, 1)):
        r, c = tok[0], tok[2]
        v = grid[r][c]
        if v in (DARKPINK, PINK):
            return ((r + c) % 2) ^ (0 if v == DARKPINK else 1) ^ flip
    return 0


def _click_portal_pair(grid, tok1, tok2):
    """Shared portal-pair click: flip the shared phase bit; if green stands on
    one portal, teleport it to the other (the vacated portal re-shows its
    pattern at the new phase). One rule for all of L3/L4/L5 magenta."""
    out = [list(row) for row in grid]
    newp = 1 - _portal_phase(grid, tok1, tok2)
    on1 = _green_on_region(grid, tok1)
    on2 = _green_on_region(grid, tok2)
    if on1 or on2:
        src, dst = (tok1, tok2) if on1 else (tok2, tok1)
        valfn = _mag_t1_val if src is tok1 else _mag_t2_val
        sr0, sr1, sc0, sc1 = src
        for r in range(sr0, sr1 + 1):
            for c in range(sc0, sc1 + 1):
                out[r][c] = valfn(r, c, newp)
        dr0, dr1, dc0, dc1 = dst
        for r in range(dr0, dr1 + 1):
            for c in range(dc0, dc1 + 1):
                out[r][c] = GREEN
        return out
    for tok, valfn in ((tok1, _mag_t1_val), (tok2, _mag_t2_val)):
        r0, r1, c0, c1 = tok
        # A token that has been RECOLOURED away from the pink/darkpink portal
        # palette (e.g. the L6 swatch-painted TOK1) no longer responds to the
        # magenta phase flip -- leave it untouched (buffer step 719: TOK1 was
        # teal/yellow and stayed put while TOK2 flipped). On L3/L4/L5 portals
        # are always pink/darkpink so this guard never changes their behaviour.
        if grid[r0][c0] not in (DARKPINK, PINK):
            continue
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r][c] = valfn(r, c, newp)
    return out


def _click_blue_l3(grid):
    """Swap the four L3 blue regions solid<->checker (same blob family)."""
    return _toggle_blue(grid, BLUE3_REGIONS)


def _click_magenta_l3(grid):
    return _click_portal_pair(grid, MAG_T1, MAG_T2)


# L5 consumable PURPLE token (buffer step 602): a purple(15) token sits in a
# gray(2) frame at rows18-19 cols10-13. Delivering green onto it CONSUMES it
# (purple cells -> gray frame) and SPAWNS a PURPLE anvil button in the right
# panel (stem rows12-14 cols49-55, base rows15-16 cols46-58) -- same unlock
# family as the L2 red / L3 white token.
PURPLE = WHITE                       # palette 15; called "purple" on L5
PURPLE_TOKEN = (18, 19, 10, 13)
PURPLE_ANVIL_STEM = (12, 14, 49, 55)
PURPLE_ANVIL_BASE = (15, 16, 46, 58)
# L5 PURPLE BLOB (buffer step 603): clicking the spawned purple anvil swaps these
# two regions solid<->checker. They sit in OPPOSITE phase (one solid, the other
# checker) and each click flips both. Solid purple is walkable floor (in _FLOOR_L5).
PURPLE5_REGIONS = ((46, 53, 26, 29), (50, 53, 14, 23))

# Per-level consumable-token data for the shared mechanic above:
#   level -> (token_box, colour, (anvil_fill_regions...))
_CONSUMABLES = {
    2: (RED_TOKEN, RED, (RED_ANVIL_STEM, RED_ANVIL_BASE)),
    3: (WHITE_TOKEN, WHITE, (WHITE_ANVIL_STEM, WHITE_ANVIL_BASE)),
    5: (PURPLE_TOKEN, PURPLE, (PURPLE_ANVIL_STEM, PURPLE_ANVIL_BASE)),
}

# L6 MULTI-COLOUR consumable token (buffer step 716): a fixed 6-cell, 4-colour
# token (red8/yellow11/lblue10/pink7) in the gray room above the left tower.
# Delivering green onto ANY of its cells consumes the whole token (cells ->
# gray frame) and REGISTERS it as a 4-colour 2x2 swatch in the right panel.
# Same consumable FAMILY as the L2 red / L3 white / L5 purple tokens, only the
# token is multi-colour and the spawned panel element is a colour swatch (not a
# single-colour anvil). Token colours include yellow/lblue which are NOT in the
# L6 floor set, so the delivery is handled explicitly (it bypasses the strict
# floor gate exactly like the win-delivery does).
LBLUE = 10
L6_TOKEN_CELLS = (((18, 7), YELLOW), ((18, 8), LBLUE),
                  ((19, 6), RED), ((19, 7), RED),
                  ((19, 8), PINK), ((19, 9), PINK))
L6_TOKEN_SWATCH = (((46, 49), RED), ((46, 50), RED),
                   ((46, 51), YELLOW), ((46, 52), YELLOW),
                   ((47, 49), RED), ((47, 50), RED),
                   ((47, 51), YELLOW), ((47, 52), YELLOW),
                   ((48, 49), LBLUE), ((48, 50), LBLUE),
                   ((48, 51), PINK), ((48, 52), PINK),
                   ((49, 49), LBLUE), ((49, 50), LBLUE),
                   ((49, 51), PINK), ((49, 52), PINK))


def _l6_token_present(grid):
    return any(grid[r][c] == col for (r, c), col in L6_TOKEN_CELLS)


def _move_lands_on_l6_token(grid, action_id):
    """True iff this move carries green onto any still-present token cell."""
    if not _l6_token_present(grid):
        return False
    dest = _green_dest(grid, action_id)
    if dest is None:
        return False
    nt, nl, nb, nr = dest
    for (r, c), col in L6_TOKEN_CELLS:
        if nt <= r <= nb and nl <= c <= nr and grid[r][c] == col:
            return True
    return False


def _consume_l6_token(grid, action_id):
    """Slide green onto the token (bypassing floor gating, since the token's
    yellow/lblue cells are not floor), revert remaining token cells to gray, and
    paint the 4-colour swatch into the right panel."""
    out = [list(row) for row in grid]
    pbb = _find_player(grid)
    dest = _green_dest(grid, action_id)
    if pbb is None or dest is None:
        return out
    pt, pl, pb, pr = pbb
    nt, nl, nb, nr = dest
    for r in range(pt, pb + 1):
        for c in range(pl, pr + 1):
            out[r][c] = _bg_fill(grid, r, c)
    for r in range(nt, nb + 1):
        for c in range(nl, nr + 1):
            out[r][c] = GREEN
    for (r, c), col in L6_TOKEN_CELLS:
        if out[r][c] == col:
            out[r][c] = FRAME
    for (r, c), col in L6_TOKEN_SWATCH:
        out[r][c] = col
    return out


# L6 greenB-room RED BLOCK consumable (buffer step 769). A red(8) block sits in
# the right-side portal room (bbox rows48-49 cols34-37). Delivering green onto it
# CONSUMES every red cell (-> gray frame) and SPAWNS a solid red anvil button in
# the right panel (rows17-19 cols47-55). Same consumable->anvil family as the L2
# red / L3 white / L5 purple tokens; red8 is walkable floor so this is checked
# BEFORE the plain move so the delivery overrides a normal walk-on.
L6_RED_BLOCK = (48, 49, 34, 37)
L6_RED_ANVIL = (17, 19, 47, 55)


def _move_lands_on_l6_red_block(grid, action_id):
    r0, r1, c0, c1 = L6_RED_BLOCK
    if not any(grid[r][c] == RED
               for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
        return False
    dest = _green_dest(grid, action_id)
    if dest is None:
        return False
    nt, nl, nb, nr = dest
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r][c] == RED and nt <= r <= nb and nl <= c <= nr:
                return True
    return False


def _consume_l6_red_block(grid, action_id):
    """Slide green onto the red block, revert ALL its red cells to gray, and
    spawn the red anvil button in the right panel (buffer step 769)."""
    out = [list(row) for row in grid]
    pbb = _find_player(grid)
    dest = _green_dest(grid, action_id)
    if pbb is None or dest is None:
        return out
    pt, pl, pb, pr = pbb
    for r in range(pt, pb + 1):
        for c in range(pl, pr + 1):
            out[r][c] = _bg_fill(grid, r, c)
    r0, r1, c0, c1 = L6_RED_BLOCK
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r][c] == RED:
                out[r][c] = FRAME
    nt, nl, nb, nr = dest
    for r in range(nt, nb + 1):
        for c in range(nl, nr + 1):
            out[r][c] = GREEN
    _fill_region(out, L6_RED_ANVIL, RED)
    return out


def _pink_in_v(grid):
    return _bar_is_v(grid, PINK, (PINK_H_LEFT, PINK_H_RIGHT),
                     (PINK_V_TOP, PINK_V_BOT))


def _click_pink(grid):
    """Toggle the L2 pink bar H<->V around its fixed darkpink pivot."""
    return _toggle_bar(grid, PINK, (PINK_H_LEFT, PINK_H_RIGHT),
                       (PINK_V_TOP, PINK_V_BOT))


def _click_blue_l2(grid):
    """Swap the two L2 blue regions solid<->checker (same family as L1)."""
    return _toggle_blue(grid, (BLUE2_BLOCK, BLUE2_CHECKER))


# --- Level 5 piece mechanics -------------------------------------------------
# L5 (discriminated by colour 10) reuses several known families plus two new
# right-panel controls. The right panel has many anvils; only those exercised
# by the buffer (steps 379-385) are modelled, dispatched by the colour AND the
# clicked component's size (the env tags each click with comp_size):
#   BLUE(9) comp 47 -> blue BLOB solid<->checker over BLUE4_REGIONS (step 379,
#       identical family/geometry to L4's blue blob).
#   BLUE(9) comp 16 -> the RED GLYPH sprite slides UP 4 rows (step 383).
#   YELLOW(11) comp 40 -> the single navy(1) bar slides RIGHT +2 cells (step 384).
#   RED(8) comp 27 -> no-op in the observed state (step 385).
#
# The RED GLYPH (buffer steps 383/399/401/402) is a rigid red(8) sprite that
# slides 4 px over a STATIC white(0) chamber on the left-board bg(4). It is the
# same slide-gated-floor idea as the green player, applied to a multi-cell rigid
# piece via a 4-button direction pad in the right panel (rows28-31):
#   leftpair colour-10 (x~44 / x~54)  -> LEFT / RIGHT
#   rightpair blue comp16 (x~50 / x~59) -> UP   / DOWN
# Move rule: shift every red cell by (dr,dc); BLOCKED (no-op) iff any destination
# cell is a WALL -- GRAY(2) or off the left board. Vacated cells revert to
# white(0) inside the chamber else bg(4); entered cells become red(8).
# The chamber is a static "C": top bar, a left channel, a bottom bar.
_L5_GLYPH_STEP = 4   # pixels the glyph moves per direction-pad click

# Static decorations the red glyph can vacate ONTO inside its roaming region.
# Everywhere else a vacated glyph cell reverts to white(0) inside the chamber
# else bg(4). These are the teal block left edge (rows14-17 cols20-21) and the
# WHITE(15) framed token (rows18-21 cols10-13, gray frame + white core).
# DEBT: the white token is treated as static; if it is ever consumed (green
# delivered onto it) before the glyph vacates these cells, the revert value
# would change. Not observed together in the buffer.
_L5_GLYPH_DECOR = {}
for _r in range(14, 18):
    for _c in (20, 21):
        if not (_r in (14, 17) and _c == 21):   # matches l5_initial teal edge
            _L5_GLYPH_DECOR[(_r, _c)] = TEAL
_L5_GLYPH_DECOR.update({
    (18, 10): FRAME, (18, 11): WHITE, (18, 12): WHITE, (18, 13): FRAME,
    (19, 10): WHITE, (19, 11): WHITE, (19, 12): WHITE, (19, 13): WHITE,
    (20, 10): FRAME, (20, 11): FRAME, (20, 12): FRAME, (20, 13): FRAME,
    (21, 10): FRAME, (21, 11): FRAME, (21, 12): FRAME, (21, 13): FRAME,
})


def _l5_glyph_revert(r, c):
    """Static value under a vacated red-glyph cell on L5."""
    if (r, c) in _L5_GLYPH_DECOR:
        return _L5_GLYPH_DECOR[(r, c)]
    # White corridor is cols >= 3; col2 is bg(4) that the core overhangs onto
    # (so it is inside _in_l5_chamber for movement, but reverts to bg here).
    return 0 if (_in_l5_chamber(r, c) and c >= 3) else LEFT_BG


def _in_l5_chamber(r, c):
    # The white(0) corridor. Left boundary is col2 (not col3): the glyph's
    # parallelogram CORE leans left, so its middle rows legitimately overhang
    # one column onto bg(4) at the corridor's left edge (buffer step 437 slides
    # the core's left edge to col2; step 418 rests there). The hard left wall is
    # enforced by the off-board check (col >= 0) in _l5_move_red_glyph.
    if 14 <= r <= 17 and 2 <= c <= 20:
        return True
    if 18 <= r <= 25 and 2 <= c <= 6:
        return True
    if 26 <= r <= 29 and 2 <= c <= 18:
        return True
    return False


def _component_size_at(grid, r, c):
    """Size of the 4-connected same-colour component containing (r, c)."""
    h, w = len(grid), len(grid[0])
    color = grid[r][c]
    seen = {(r, c)}
    stack = [(r, c)]
    while stack:
        rr, cc = stack.pop()
        for nr, nc in ((rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen \
                    and grid[nr][nc] == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen)


def _l5_move_red_glyph(grid, dr, dc):
    """Direction-pad anvil: slide the rigid left-board red(8) glyph 4 px by
    (dr,dc) over the static white(0) chamber. The glyph is an I-beam/H shape:
    a dense 4-row CORE (its middle rows) flanked by thin top/bottom crossbar
    ARMS. The CORE must stay seated on the white(0) corridor; the ARMS may
    overhang freely into open background. The glyph slides over gray(2),
    white(0) and bg(4) alike -- gray is NOT a wall (buffer step 430 slides the
    glyph onto a gray cell). A move is a no-op iff:
      * any destination cell is off the left board, OR
      * any CORE cell that currently sits on the white corridor would move off
        the corridor (you cannot push the body through a corridor wall).
    Buffer (all 18 glyph clicks 383-431): move 383/399/401/403/404/416/417/
    418/430/431; no-op 392/398/400/402/405/406/415 (off-board or core leaves
    chamber)."""
    h = len(grid)
    red = [(r, c) for r in range(h) for c in range(RIGHT_X)
           if grid[r][c] == RED]
    if not red:
        return grid
    minr = min(r for (r, c) in red)
    shifted = [(r + dr, c + dc) for (r, c) in red]
    for (r, c) in shifted:
        if not (0 <= r < h and 0 <= c < RIGHT_X):
            return grid  # off the left board: blocked
    # CORE = the dense middle 4 rows (rel rows 2..5 of the 8-row glyph). The
    # glyph slides FREELY over gray(2)/white(0)/bg(4); the only interior
    # constraint is that its CORE stays seated on the white(0) chamber (the
    # thin top/bottom crossbar ARMS may overhang into background).
    for (r, c) in red:
        if minr + 2 <= r <= minr + 5 and _in_l5_chamber(r, c) \
                and not _in_l5_chamber(r + dr, c + dc):
            return grid  # core would leave the corridor: blocked
    out = [list(row) for row in grid]
    for (r, c) in red:
        out[r][c] = _l5_glyph_revert(r, c)
    for (r, c) in shifted:
        out[r][c] = RED
    return out


# Direction-pad button columns -> glyph (dr, dc). The pad is a single row-band
# at rows28-31; horizontal pair is colour-10, vertical pair is blue comp16.
_L5_GLYPH_DIRS = {
    (43, 46): (0, -_L5_GLYPH_STEP),   # LEFT  (colour-10, left button)
    (47, 50): (-_L5_GLYPH_STEP, 0),   # UP    (blue comp16, left button)
    (53, 56): (0, _L5_GLYPH_STEP),    # RIGHT (colour-10, right button)
    (57, 60): (_L5_GLYPH_STEP, 0),    # DOWN  (blue comp16, right button)
}


def _l5_glyph_dir(x):
    for (lo, hi), d in _L5_GLYPH_DIRS.items():
        if lo <= x <= hi:
            return d
    return None


# The red glyph is a toggle-shape sprite (same family as the L1-L3 red bar
# H<->V): EXTENDED (8 rows tall, 30 cells) <-> CONTRACTED (6 rows tall, 28
# cells). Both bitmaps share the same left column and vertical centre; the RED
# anvil toggles between them but ONLY settles when the glyph abuts the TEAL
# bar on its right (buffer: settle at step460, refused at 385/409/433/435/439/
# 444/447 -- all non-teal-adjacent). Once contracted the glyph MESHES with the
# teal notched edge and DRAGS the teal bar with it on horizontal translation
# (buffer steps 461 LEFT, 463 RIGHT). Teal is walkable floor -> a movable
# bridge. DEBT: extend-back direction and any vertical teal drag are unobserved.
_L5_GLYPH_EXT = [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 3), (1, 7),
                 (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2),
                 (3, 3), (4, 0), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2),
                 (5, 3), (5, 4), (6, 3), (6, 7), (7, 3), (7, 4), (7, 5),
                 (7, 6), (7, 7)]
_L5_GLYPH_CON = [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 1), (1, 2),
                 (1, 3), (1, 4), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3),
                 (3, 0), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3),
                 (4, 4), (4, 7), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)]


def _l5_glyph_cells(grid):
    """Red glyph cells + bbox (minr, maxr, minc, maxc), or None."""
    red = [(r, c) for r in range(40) for c in range(RIGHT_X)
           if grid[r][c] == RED]
    if not red:
        return None
    minr = min(r for r, c in red)
    maxr = max(r for r, c in red)
    minc = min(c for r, c in red)
    maxc = max(c for r, c in red)
    return red, minr, maxr, minc, maxc


def _l5_teal_bar_cells(grid):
    """Teal(12) cells of the movable bar (rows<44); excludes the
    rows44-45 transfer span which is also teal when full. The bar can be
    dragged down by the contracted glyph (buffer steps 488-490 put it at
    rows22-33), so the cutoff must clear row33 yet stay above the span."""
    return [(r, c) for r in range(44) for c in range(RIGHT_X)
            if grid[r][c] == TEAL]


def _l5_plain_revert(r, c):
    """Static value under a vacated glyph/teal-bar cell, WITHOUT the teal
    decoration (teal is rendered dynamically when the bar is movable): white(0)
    inside the chamber, the white-token frame decor, else bg(4)."""
    v = _L5_GLYPH_DECOR.get((r, c))
    if v is not None and v != TEAL:
        return v
    return 0 if (_in_l5_chamber(r, c) and c >= 3) else LEFT_BG


def _l5_glyph_contracted(grid):
    info = _l5_glyph_cells(grid)
    return info is not None and (info[2] - info[1] + 1) == 6


def _click_red_l5(grid):
    """RED anvil: CONTRACT the glyph (EXTENDED -> CONTRACTED), but only when the
    glyph is currently extended AND abuts the teal bar on its right (else no-op).
    The reverse (contracted -> extended) is never settled: a contracted glyph
    meshed against the teal bar cannot grow back (buffer step466 = no-op; step460
    is the only observed settle, EXT->CON). All 8 red clicks (385/409/433/435/
    439/444 refused off-teal; 460 settled EXT->CON; 466 refused CON-on-teal)."""
    info = _l5_glyph_cells(grid)
    if info is None:
        return grid
    red, minr, maxr, minc, maxc = info
    if maxr - minr + 1 != 8:          # only the extended glyph can contract
        return grid
    teal = set(_l5_teal_bar_cells(grid))
    if not any((r, maxc + 1) in teal for r in range(minr, maxr + 1)):
        return grid
    out = [list(row) for row in grid]
    for (r, c) in red:
        out[r][c] = _l5_glyph_revert(r, c)
    base = minr + 1                   # pull in 1 row top & bottom
    for (dr, dc) in _L5_GLYPH_CON:
        out[base + dr][minc + dc] = RED
    return out


def _l5_move_contracted(grid, dr, dc):
    """Translate the CONTRACTED glyph by (dr,dc) AND drag the meshed teal bar
    with it. Same off-board / core-on-chamber block rule as the extended glyph;
    vacated cells revert to white-chamber/bg (teal is re-rendered at +d)."""
    info = _l5_glyph_cells(grid)
    if info is None:
        return grid
    red, minr, maxr, minc, maxc = info
    teal = _l5_teal_bar_cells(grid)
    # Toggle-locked-under-player: refuse the drag if green rides the teal bar
    # (cannot pull the bridge out from under the token). Buffer step 497: a
    # contracted drag with green inside the bar's bbox is a net no-op.
    if teal:
        tminr = min(r for r, _ in teal); tmaxr = max(r for r, _ in teal)
        tminc = min(c for _, c in teal); tmaxc = max(c for _, c in teal)
        for r in range(len(grid)):
            for c in range(RIGHT_X):
                if grid[r][c] == GREEN and tminr <= r <= tmaxr \
                        and tminc <= c <= tmaxc:
                    return grid
    shifted = [(r + dr, c + dc) for (r, c) in red]
    for (r, c) in shifted:
        if not (0 <= r < len(grid) and 0 <= c < RIGHT_X):
            return grid
    for (r, c) in red:
        if minr + 2 <= r <= maxr - 2 and _in_l5_chamber(r, c) \
                and not _in_l5_chamber(r + dr, c + dc):
            return grid
    out = [list(row) for row in grid]
    for (r, c) in red:
        out[r][c] = _l5_plain_revert(r, c)
    for (r, c) in teal:
        out[r][c] = _l5_plain_revert(r, c)
    for (r, c) in teal:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < RIGHT_X:
            out[nr][nc] = TEAL
    for (r, c) in shifted:
        out[r][c] = RED
    return out


# L5 DARKPINK anvil: the two 2x2 pink/darkpink tokens are a TELEPORT portal
# pair (same family as L3/L4 magenta). P1 sits in the upper region, P2 in the
# small lower-left region. Every press flips the shared checker phase bit; if
# green stands on one portal the press teleports it to the other (buffer step
# 386 observed the plain phase flip with green off both portals).
L5_TOK1 = (6, 7, 24, 25)
L5_TOK2 = (34, 35, 10, 11)


def _click_magenta_l5(grid):
    return _click_portal_pair(grid, L5_TOK1, L5_TOK2)


# ---- L6 magenta(darkpink) anvil (buffer step 666) ----
# A single click does THREE coupled things, modelled together:
#  1. Rotate a TEAL I-beam bar H<->V around a FIXED red(8) pivot ring
#     (rows32-35 cols8-11). The bar passes through the hollow pivot centre, so
#     those centre cells stay teal in both orientations. Underneath the right
#     H-arm is void(0) (rows32-35 cols14-19); everywhere else is bg(4).
#  2. Teleport green between the two darkpink/pink portals (portal-pair family).
#  3. Flip two small phase indicators (a yellow/teal 2x2 and a blue/lblue 2x2).
L6_TOK1 = (48, 49, 18, 19)
L6_TOK2 = (52, 53, 32, 33)
_L6_TEAL_H = frozenset(((30, 7), (30, 8), (30, 9), (30, 10), (30, 11), (30, 12), (31, 8), (31, 9), (31, 10), (31, 11), (32, 0), (32, 1), (32, 2), (32, 3), (32, 4), (32, 5), (32, 6), (32, 7), (32, 12), (32, 13), (32, 14), (32, 15), (32, 16), (32, 17), (32, 18), (32, 19), (33, 0), (33, 1), (33, 2), (33, 3), (33, 4), (33, 5), (33, 6), (33, 7), (33, 9), (33, 10), (33, 12), (33, 13), (33, 14), (33, 15), (33, 16), (33, 17), (33, 18), (33, 19), (34, 0), (34, 1), (34, 2), (34, 3), (34, 4), (34, 5), (34, 6), (34, 7), (34, 9), (34, 10), (34, 12), (34, 13), (34, 14), (34, 15), (34, 16), (34, 17), (34, 18), (34, 19), (35, 0), (35, 1), (35, 2), (35, 3), (35, 4), (35, 5), (35, 6), (35, 7), (35, 12), (35, 13), (35, 14), (35, 15), (35, 16), (35, 17), (35, 18), (35, 19), (36, 8), (36, 9), (36, 10), (36, 11), (37, 7), (37, 8), (37, 9), (37, 10), (37, 11), (37, 12)))
_L6_TEAL_V = frozenset(((24, 8), (24, 9), (24, 10), (24, 11), (25, 8), (25, 9), (25, 10), (25, 11), (26, 8), (26, 9), (26, 10), (26, 11), (27, 8), (27, 9), (27, 10), (27, 11), (28, 8), (28, 9), (28, 10), (28, 11), (29, 8), (29, 9), (29, 10), (29, 11), (30, 8), (30, 9), (30, 10), (30, 11), (31, 6), (31, 8), (31, 9), (31, 10), (31, 11), (31, 13), (32, 6), (32, 7), (32, 12), (32, 13), (33, 6), (33, 7), (33, 9), (33, 10), (33, 12), (33, 13), (34, 6), (34, 7), (34, 9), (34, 10), (34, 12), (34, 13), (35, 6), (35, 7), (35, 12), (35, 13), (36, 6), (36, 8), (36, 9), (36, 10), (36, 11), (36, 13), (37, 8), (37, 9), (37, 10), (37, 11), (38, 8), (38, 9), (38, 10), (38, 11), (39, 8), (39, 9), (39, 10), (39, 11), (40, 8), (40, 9), (40, 10), (40, 11), (41, 8), (41, 9), (41, 10), (41, 11), (42, 8), (42, 9), (42, 10), (42, 11), (43, 8), (43, 9), (43, 10), (43, 11)))


def _toggle_teal_bar_l6(out):
    """Rotate the L6 teal I-beam H<->V around the fixed red pivot ring."""
    cur, tgt = ((_L6_TEAL_H, _L6_TEAL_V) if out[32][0] == TEAL
                else (_L6_TEAL_V, _L6_TEAL_H))
    for (r, c) in cur - tgt:
        out[r][c] = 0 if (32 <= r <= 35 and 14 <= c <= 19) else 4
    for (r, c) in tgt:
        out[r][c] = TEAL


def _swap_l6_indicators(out):
    """Flip the two small L6 phase indicators on a magenta press (used only by
    the UNPAINTED default-portal branch of _click_magenta_l6)."""
    for (r0, r1, c0, c1, ca, cb) in ((4, 5, 4, 5, 11, 12), (58, 59, 34, 35, 9, 10)):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r][c] == ca:
                    out[r][c] = cb
                elif out[r][c] == cb:
                    out[r][c] = ca


# L6 PORTAL STATIONS. The painted HOME token TOK1 (rows48-49 cols18-19)
# teleports green to the colour-MATCHED destination station on a magenta press
# (buffer steps 759/762). Three stations exist, each a 2x2 (dark,light) checker
# sharing ONE global phase bit that flips on every magenta press:
#   A    (4-5,4-5)     teal/yellow  -- matches a TEAL/YELLOW paint
#   B    (58-59,34-35) blue/lblue   -- matches a BLUE/LBLUE paint
#   TOK2 (52-53,32-33) darkpink/pink-- the default portal partner
# Earlier runs mislabelled A and B as inert "phase indicators"; they are the
# teleport endpoints selected by TOK1's paint colour.
_L6_STATION_A = (4, 5, 4, 5)
_L6_STATION_B = (58, 59, 34, 35)
_L6_STATIONS = (
    (_L6_STATION_A, (TEAL, YELLOW)),
    (_L6_STATION_B, (BLUE, LBLUE)),
    (L6_TOK2, (DARKPINK, PINK)),
)
# Paint pairs the swatch can apply (dark, light). darkpink/pink == the default
# portal palette, so a pink-painted TOK1 is treated as the plain portal (None).
_L6_PAINT_PAIRS = ((TEAL, YELLOW), (BLUE, LBLUE), (DARKRED, RED))
_L6_TOK1_PAINT_MEM = None


def _l6_tok1_paint(grid):
    """TOK1's (dark, light) paint pair, or None when it is the default
    darkpink/pink portal. Reconstructed from the frame whenever TOK1 is visible;
    when green sits on TOK1 (paint hidden under the player) fall back to the
    remembered value last set by a swatch paint or a visible frame. The paint
    selects which station a HOME magenta press teleports green to."""
    global _L6_TOK1_PAINT_MEM
    r0, r1, c0, c1 = L6_TOK1
    if grid[r0][c0] != GREEN:
        cells = {grid[r][c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if cells <= {DARKPINK, PINK}:
            _L6_TOK1_PAINT_MEM = None
            return None
        for pair in _L6_PAINT_PAIRS:
            if cells <= set(pair):
                _L6_TOK1_PAINT_MEM = pair
                return pair
    return _L6_TOK1_PAINT_MEM


def _l6_global_phase(grid):
    """Shared checker-phase bit of the L6 stations (all share one parity, read
    from the first visible station). Phase p: cell(r,c)=dark iff (r+c)%2==p."""
    for region, (d, l) in _L6_STATIONS:
        r0, _, c0, _ = region
        v = grid[r0][c0]
        if v == d:
            return (r0 + c0) % 2
        if v == l:
            return 1 - (r0 + c0) % 2
    return 0


def _render_checker(out, region, dark, light, phase):
    r0, r1, c0, c1 = region
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r][c] = dark if (r + c) % 2 == phase else light


def _click_magenta_l6(grid):
    """L6 magenta(darkpink) anvil. Rotates the teal I-beam and drives the portal
    system. While TOK1 is the default darkpink/pink portal this is the plain
    TOK1<->TOK2 teleport + phase flip (early L6). Once TOK1 has been swatch-
    painted it is a colour-keyed HOME: a press flips every station's shared
    phase, and if green stands on an endpoint it teleports -- HOME->station whose
    colour matches TOK1's paint (A teal/yellow, B blue/lblue), or station->HOME
    (buffer steps 759/760/762). HOME always renders its fixed paint (dark on
    even); the destination becomes green; a station green just left re-renders at
    the new shared phase."""
    global _L6_TOK1_PAINT_MEM
    paint = _l6_tok1_paint(grid)
    if paint is None:
        out = _click_portal_pair(grid, L6_TOK1, L6_TOK2)
        _toggle_teal_bar_l6(out)
        _swap_l6_indicators(out)
        return out
    out = [list(row) for row in grid]
    _toggle_teal_bar_l6(out)
    newp = 1 - _l6_global_phase(grid)
    on_home = _green_on_region(grid, L6_TOK1)
    src_station = None
    for region, _pair in _L6_STATIONS:
        if _green_on_region(grid, region):
            src_station = region
            break
    for region, (d, l) in _L6_STATIONS:
        _render_checker(out, region, d, l, newp)
    dest_station = None
    for region, pair in _L6_STATIONS:
        if pair == paint:
            dest_station = region
            break
    if dest_station is not None:
        # Painted a STATION colour: HOME keeps its paint (rendered at fixed
        # phase 0) and the press teleports green to the colour-matched station.
        pd, pl = paint
        _render_checker(out, L6_TOK1, pd, pl, 0)
        if on_home:
            _fill_region(out, dest_station, GREEN)
        elif src_station is not None:
            _fill_region(out, L6_TOK1, GREEN)
    else:
        # Painted a NON-STATION colour (red has no station, buffer step 766):
        # TOK1 falls back to the DEFAULT darkpink/pink portal, so the press is
        # the plain HOME<->TOK2 teleport. The paint is dropped (TOK1 re-renders
        # default at the new shared phase) and the memory is cleared.
        r0, r1, c0, c1 = L6_TOK1
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r][c] = _mag_t1_val(r, c, newp)
        _L6_TOK1_PAINT_MEM = None
        if on_home:
            _fill_region(out, L6_TOK2, GREEN)
        elif src_station is L6_TOK2:
            _fill_region(out, L6_TOK1, GREEN)
    return out


# L6 COLOUR SWATCH (buffer steps 716-718). Consuming the multi-colour token
# registers a 2x2-quadrant swatch in the right panel (rows46-49 cols49-52:
# red TL, yellow TR, lblue BL, pink BR). Clicking a swatch quadrant PAINTS the
# portal token TOK1 (rows48-49 cols18-19) with that colour's (dark, light)
# checker -- (r+c) even -> dark shade, odd -> light shade.
_L6_SWATCH_REGION = (46, 49, 49, 52)         # rows46-49 cols49-52
_L6_SWATCH_PAIRS = {                          # clicked colour -> (dark, light)
    RED: (DARKRED, RED),                      # 8  -> (13, 8)
    YELLOW: (TEAL, YELLOW),                   # 11 -> (12, 11)
    10: (BLUE, 10),                           # lblue -> (9, 10)
    PINK: (DARKPINK, PINK),                   # 7  -> (6, 7)
}


def _click_l6_swatch(grid, val):
    """Paint TOK1 with the clicked swatch colour pair. When green stands ON TOK1
    the repaint is hidden under the player so the board is unchanged, but the
    underlying paint is still updated (buffer step 761: the lblue swatch click is
    a visual no-op, yet step 762's HOME press teleports to the blue/lblue
    station -- the selection registered under the player)."""
    global _L6_TOK1_PAINT_MEM
    dark, light = _L6_SWATCH_PAIRS[val]
    _L6_TOK1_PAINT_MEM = None if (dark, light) == (DARKPINK, PINK) else (dark, light)
    out = [list(row) for row in grid]
    if _green_on_region(grid, L6_TOK1):
        return out
    _render_checker(out, L6_TOK1, dark, light, 0)
    return out


_L6_NAVY_LANES = (54, 56, 58)   # top row of each 2-row conveyor lane
_L6_NAVY_ENTRY = 6              # leftmost col of phase 0
_L6_NAVY_DOCK = 16             # leftmost col of phase 5 (rendered TEAL)
_L6_NAVY_W = 6                 # bar width
_L6_NAVY_PERIOD = 6            # phases 0..5 (6,8,10,12,14 navy; 16 teal)


def _l6_click_strands_green(grid):
    """True iff a conveyor click would slide the bar green rides out from under
    it. Green covers <=2 cells of a 6-wide bar; advancing the bar +2px (or
    wrapping from the dock) can leave green off the new span -> the env refuses
    the whole click (the same toggle-locked-under-player rule all L6/L4 bar
    cycles obey). Buffer step 681 (green on a docked teal bar -> wrap strands),
    step 690 (green on a navy bar at leftmost 14 -> dock-slide strands)."""
    lanes = _L6_NAVY_LANES
    for r in lanes:
        rows = (r, r + 1)
        gcols = set()
        bcols = set()
        for rr in rows:
            for c in range(4, 28):
                v = grid[rr][c]
                if v == GREEN:
                    gcols.add(c)
                elif v in (NAVY, TEAL):
                    bcols.add(c)
        if not gcols or not bcols:
            continue
        mat = sorted(gcols | bcols)
        # maximal runs of bar+green material
        runs = []
        s = p = mat[0]
        for c in mat[1:]:
            if c == p + 1:
                p = c
            else:
                runs.append((s, p))
                s = p = c
        runs.append((s, p))
        gl, gr = min(gcols), max(gcols)
        for L, R in runs:
            if not (L <= gl and gr <= R):
                continue           # green not in this run
            if not (bcols & set(range(L, R + 1))):
                continue           # run is all green (green not actually on a bar)
            # bars sit at the run's two ends; green rides the one whose span
            # contains it.
            bars = [L]
            if R - 5 > L:
                bars.append(R - 5)
            for b in bars:
                if b <= gl and gr <= b + 5:
                    ph = ((b - _L6_NAVY_ENTRY) // 2 + 1) % _L6_NAVY_PERIOD
                    nl = _L6_NAVY_ENTRY + 2 * ph
                    if not (nl <= gl and gr <= nl + 5):
                        return True
    return False


def _click_yellow_l6(grid):
    """L6 navy anvil (buffer steps 669,672,675,676,680): THREE INDEPENDENT
    period-6 conveyor lanes (top rows 54/56/58, each 2 rows tall). Each lane
    holds one or more 6-wide bars; every click advances every bar one phase
    right (+2px). Phases run leftmost 6,8,10,12,14 (NAVY) then 16 (TEAL dock);
    advancing from the dock WRAPS the bar back to the entry phase (leftmost 6)
    as navy. Bars within a lane overlap (navy renders over teal). The "snake"
    interpretation of earlier runs was wrong -- the lanes are independent.
    No-op when no navy/teal bar is present so it never fires on unrelated
    frames. DEBT: a run is decomposed into bars only at its two ends, so a lane
    with 3+ bars whose middle one differs in wrap state could mispredict; only
    1 (lanes 54/58) or 2 (lane 56) bars are ever observed."""
    w = len(grid[0])
    lanes = _L6_NAVY_LANES
    if not any(grid[r][c] in (NAVY, TEAL)
               for r in lanes for c in range(4, 28)):
        return grid
    # TOGGLE-LOCKED-UNDER-PLAYER: if green is riding a bar and the advance would
    # move that bar out from under green (green no longer on the new span), the
    # whole conveyor click is refused -- a pure no-op (buffer steps 681, 690).
    if _l6_click_strands_green(grid):
        return grid
    out = [list(row) for row in grid]
    for r in lanes:
        rows = (r, r + 1)
        mat = set()
        for rr in rows:
            for c in range(4, 28):
                if grid[rr][c] in (NAVY, TEAL):
                    mat.add(c)
        # green riding a bar counts as bar material (so a green-occluded bar is
        # reconstructed at full width and slides with the bar underneath).
        for rr in rows:
            for c in range(4, 28):
                if grid[rr][c] == GREEN and (
                        any(cc in mat for cc in range(max(4, c - 6), c))
                        or any(cc in mat for cc in range(c + 1, min(28, c + 7)))):
                    mat.add(c)
        if not mat:
            continue
        # maximal runs -> each run yields a bar at its left end and, if wider
        # than one bar, a bar at its right end (R-5).
        cols = sorted(mat)
        runs = []
        s = p = cols[0]
        for c in cols[1:]:
            if c == p + 1:
                p = c
            else:
                runs.append((s, p))
                s = p = c
        runs.append((s, p))
        bars = []
        for L, R in runs:
            bars.append(L)
            if R - 5 > L:
                bars.append(R - 5)
        # clear this lane's bar material (green stays in place)
        for rr in rows:
            for c in range(4, 28):
                if out[rr][c] in (NAVY, TEAL):
                    out[rr][c] = LEFT_BG
        # advance each bar one phase and render (teal first, navy on top)
        newbars = []
        for L in bars:
            ph = ((L - _L6_NAVY_ENTRY) // 2 + 1) % _L6_NAVY_PERIOD
            nl = _L6_NAVY_ENTRY + 2 * ph
            newbars.append((nl, TEAL if nl >= _L6_NAVY_DOCK else NAVY))
        for want in (TEAL, NAVY):
            for nl, color in newbars:
                if color != want:
                    continue
                for rr in rows:
                    for c in range(nl, nl + _L6_NAVY_W):
                        if 0 <= c < w and out[rr][c] != GREEN:
                            out[rr][c] = color
    return out


_L5_NAVY_ENTRY_LEFT = 14   # entry span = cols 14..19
_L5_NAVY_DOCK_LEFT = 24    # the slide that lands leftmost==24 DOCKS (teal c24-29)


def _l5_slide_navy(grid):
    """YELLOW anvil: the single left-board navy(1) BAR (rows54-55) cycles RIGHT
    with PERIOD 6, mirroring the L4 navy-bar cycle (`_click_yellow_l4`):
      - entry span cols14-19; each click slides the 6-wide bar +2px right
        (leftmost 14->16->18->20->22; buffer steps 607-609);
      - the slide from leftmost 22 DOCKS the bar at cols24-29 and converts it to
        TEAL(12), the docked colour (buffer step 625);
      - the next click (from the docked TEAL state) WRAPS the bar back to the
        entry navy span cols14-19 (buffer step 626).
    Restricted to the bar's own rows so the reservoir-span navy/teal ends
    (rows44-45, present once the gray pump partially fills) are never dragged."""
    out = [list(row) for row in grid]
    rows = _L5_NAVY_BAR_ROWS
    w = len(grid[0])
    # DOCKED state: teal in a bar row -> WRAP both rows back to entry navy.
    docked = any(grid[r][c] == TEAL for r in rows for c in range(w))
    if docked:
        for r in rows:
            for c in range(w):
                if grid[r][c] == TEAL:
                    out[r][c] = LEFT_BG
            for c in range(_L5_NAVY_ENTRY_LEFT, _L5_NAVY_ENTRY_LEFT + 6):
                if grid[r][c] != GREEN:
                    out[r][c] = NAVY
        return out
    # SLIDE: recover the 6-wide bar span (green may occlude up to 2 cells),
    # move +2; convert to TEAL when the new leftmost reaches the dock column.
    navy = [c for c in range(w) if grid[rows[0]][c] == NAVY
            or grid[rows[1]][c] == NAVY]
    if not navy:
        return out
    L, R = min(navy), max(navy)
    if R - L + 1 < 6:                      # part of bar hidden under green
        if R + 1 < w and grid[rows[0]][R + 1] == GREEN:
            R = L + 5
        elif L - 1 >= 0 and grid[rows[0]][L - 1] == GREEN:
            L = R - 5
        else:
            R = L + 5
    new_left = L + 2
    color = TEAL if new_left >= _L5_NAVY_DOCK_LEFT else NAVY
    for r in rows:
        for c in range(L, R + 1):
            if grid[r][c] == NAVY:
                out[r][c] = LEFT_BG
        for c in range(new_left, new_left + 6):
            if 0 <= c < w and grid[r][c] != GREEN:
                out[r][c] = color
    return out


# --- Public contract ---------------------------------------------------------

def transition_function(frame, action_id):
    """Predict next frame from (frame, action)."""
    grid = [list(row) for row in frame]

    x = y = None
    if isinstance(action_id, dict):
        aid = action_id.get("action_id")
        x = action_id.get("x")
        y = action_id.get("y")
    else:
        aid = action_id

    # Keep the L6 portal-paint memory current from any frame where TOK1 is
    # visible, so a HOME magenta press can still resolve the teleport target
    # while green hides the paint (buffer steps 759/762).
    if _detect_level(grid) == 6:
        _l6_tok1_paint(grid)

    if aid == 0:  # RESET -> CURRENT level's entry frame (cache); not a global
        # reset to L1. Buffer step 292 (RESET on L4) restored the L4 entry
        # frame, not L1. Reload this level's l<N>_initial.pkl.
        cached = load_level_frame("l%d_initial.pkl" % _detect_level(grid))
        if cached is not None:
            return [list(row) for row in cached]
        return grid

    if aid in (A_UP, A_DOWN, A_LEFT, A_RIGHT):
        # L6 multi-colour consumable token: delivering green onto it consumes
        # the token and registers its 4-colour swatch in the right panel. Checked
        # BEFORE the win predicate because the token carries a single yellow cell
        # that _find_yellow could otherwise mistake for the delivery target.
        if _detect_level(grid) == 6 and _move_lands_on_l6_token(grid, aid):
            return _consume_l6_token(grid, aid)
        # L6 greenB-room red block: delivering green onto it consumes the block
        # and spawns the red anvil (buffer step 769). red8 is floor, so this
        # must override the plain walk-on move.
        if _detect_level(grid) == 6 and _move_lands_on_l6_red_block(grid, aid):
            return _consume_l6_red_block(grid, aid)
        if _is_winning_move(grid, aid):
            # Delivering green onto the yellow target completes the level ->
            # the env loads the NEXT level's entry frame (uncomputable; cache).
            nxt = load_level_frame("l%d_initial.pkl" % (_detect_level(grid) + 1))
            if nxt is not None:
                return [list(row) for row in nxt]
        # Shared consumable-token delivery (L2 red, L3 white): green delivered
        # onto a framed anvil token consumes it and spawns its anvil button.
        spec = _CONSUMABLES.get(_detect_level(grid))
        if spec is not None:
            box, color, anvils = spec
            if _move_lands_on_token(grid, aid, box, color):
                return _consume_token(grid, aid, box, color, anvils)
        return _move_player(grid, aid)

    if aid == 6 and x is not None and y is not None:
        # Click in the RIGHT preview panel toggles that colour's piece. The
        # trigger-button colours and the affected geometry differ per level.
        if 0 <= y < len(grid) and 34 <= x < len(grid[0]):
            val = grid[y][x]
            lvl = _detect_level(grid)
            if lvl == 6:
                # COLOUR SWATCH click (buffer steps 717-718): clicking a swatch
                # quadrant (rows46-49 cols49-52) paints the portal token TOK1
                # with that colour's dark/light checker. Checked first so the
                # small red/yellow swatch buttons aren't mistaken for the
                # conveyor / other anvils.
                sr0, sr1, sc0, sc1 = _L6_SWATCH_REGION
                if (sr0 <= y <= sr1 and sc0 <= x <= sc1
                        and val in _L6_SWATCH_PAIRS):
                    return _click_l6_swatch(grid, val)
                # L6 magenta(darkpink) anvil: teal-bar rotate + portal teleport
                # + phase-indicator flip (buffer step 666). Other L6 anvils
                # remain unobserved -> no-op.
                if val == DARKPINK:
                    return _click_magenta_l6(grid)
                # Two YELLOW anvils exist: the large navy-conveyor button
                # (comp~40, buffer 55,8) drives the slide; a small 2x2 yellow
                # indicator (comp 4, buffer 46,6) is a no-op.
                if val == YELLOW and _component_size_at(grid, y, x) >= 20:
                    return _click_yellow_l6(grid)
                return grid
            if lvl == 5:
                # L5 right-panel anvils, dispatched by colour + component size
                # (the env distinguishes the two blue anvils by their size).
                # The glyph direction-pad sits at rows28-31: colour-10 buttons
                # move the glyph horizontally, blue comp16 buttons vertically.
                if 28 <= y <= 31 and val in (BLUE, 10):
                    d = _l5_glyph_dir(x)
                    if d is not None and _component_size_at(grid, y, x) <= 16:
                        if _l5_glyph_contracted(grid):
                            return _l5_move_contracted(grid, d[0], d[1])
                        return _l5_move_red_glyph(grid, d[0], d[1])
                if val == RED:          # red anvil -> glyph contract/extend
                    return _click_red_l5(grid)
                if val == BLUE:
                    return _toggle_blue(grid, BLUE4_REGIONS)
                if val == YELLOW:
                    return _l5_slide_navy(grid)
                if val == FRAME:        # gray block -> fill the L5 transfer span
                    return _click_gray_l5(grid)
                if val == DARKPINK:     # darkpink anvil -> portal-pair phase flip
                    return _click_magenta_l5(grid)
                if val == PURPLE:       # purple anvil -> purple blob solid<->checker
                    return _toggle_blue(grid, PURPLE5_REGIONS, color=PURPLE)
                return grid
            if lvl == 4:
                # L4 right-panel anvils, dispatched by the colour under click.
                if val == YELLOW:       # yellow anvil -> slide navy bars right
                    return _click_yellow_l4(grid)
                if val == DARKPINK:     # darkpink anvil -> flip pink-token phase
                    return _click_magenta_l4(grid)
                if val == FRAME:        # gray block -> void/teal coupled toggle
                    return _click_gray_l4(grid)
                if val == BLUE:         # blue anvil -> blue blob solid<->checker
                    return _toggle_blue(grid, BLUE4_REGIONS)
                return grid
            if lvl == 3:
                if val == RED:          # red anvil -> red bar H<->V (RED2 geom)
                    return _click_red_l2(grid)
                if val == BLUE:         # blue anvil -> swap four blue regions
                    return _click_blue_l3(grid)
                if val == DARKPINK:     # magenta anvil -> flip both 6/7 tokens
                    return _click_magenta_l3(grid)
                if val == WHITE:        # white anvil -> white blob solid<->checker
                    return _click_white_l3(grid)
                return grid
            if lvl == 2:
                if val == DARKPINK:        # magenta anvil -> pink bar H<->V
                    return _click_pink(grid)
                if val == BLUE:            # blue anvil -> swap blue phases
                    return _click_blue_l2(grid)
                if val == RED:             # unlocked red anvil -> red bar H<->V
                    return _click_red_l2(grid)
                return grid
            if val == 0:  # landed on a white outline -> pick by row
                val = RED if y < 28 else BLUE
            if val == RED:
                return _click_red(grid)
            if val == BLUE:
                return _click_blue(grid)
        return grid  # left-panel / empty click: no-op

    # ACTION7 (undo) and anything else: board unchanged.
    return grid


def _blob_solid(grid):
    """True when the blue blob bridge (BLUE_B) is fully solid (phase B) -- the
    state in which green can climb through it."""
    return _region_full(grid, BLUE_B, BLUE)


def reward_function(frame, action_id, next_frame):
    """CONFIRMED goal: deliver the GREEN token onto the YELLOW target token
    (the inert framed token at rows20-21, cols24-25). The level is won the
    instant a move would carry green onto the yellow block.

    Observed (step 131, ACTION4): green at rows20-21 cols22-23 presses RIGHT,
    lands on the yellow token -> reward 1.0, done, level advances to L2. The
    precondition chain (climb left corridor -> traverse the red=H band strip to
    the darkred base -> toggle red=V to bridge rows24-29 -> make blue-A solid ->
    climb the right corridor to the yellow token) is exactly what the buffer
    walked through at 0 reward until the final delivery move."""
    aid = action_id.get("action_id") if isinstance(action_id, dict) else action_id
    if aid in (A_UP, A_DOWN, A_LEFT, A_RIGHT) and _is_winning_move(frame, aid):
        return (1.0, True)
    return (0.0, False)


# Canonical right-panel click coordinates (display space) for the planner.
RED_PREVIEW_CLICK = {"action_id": 6, "x": 48, "y": 20}
BLUE_PREVIEW_CLICK = {"action_id": 6, "x": 48, "y": 37}
# L2 anvil-button clicks (magenta=pink toggle, blue=blue-phase, red=red bar once
# the red token is consumed and the red anvil has spawned).
PINK_ANVIL_CLICK = {"action_id": 6, "x": 52, "y": 22}
BLUE2_ANVIL_CLICK = {"action_id": 6, "x": 52, "y": 40}
RED2_ANVIL_CLICK = {"action_id": 6, "x": 52, "y": 31}
# L3 anvil-button clicks (red bar H/V, blue four-region phase, magenta token swap)
# from buffer steps 206-208: red=(51,18), blue=(51,27), magenta=(51,36).
RED3_ANVIL_CLICK = {"action_id": 6, "x": 51, "y": 18}
BLUE3_ANVIL_CLICK = {"action_id": 6, "x": 51, "y": 27}
MAG3_ANVIL_CLICK = {"action_id": 6, "x": 51, "y": 36}
# White anvil button (appears after the white token is consumed; buffer step 243).
WHITE3_ANVIL_CLICK = {"action_id": 6, "x": 51, "y": 44}
# L4 anvil-button clicks (buffer steps 259-261): yellow anvil slides navy bars,
# darkpink anvil flips pink tokens, gray block toggles the void/teal coupling.
YELLOW4_ANVIL_CLICK = {"action_id": 6, "x": 47, "y": 28}
MAG4_ANVIL_CLICK = {"action_id": 6, "x": 52, "y": 20}
GRAY4_ANVIL_CLICK = {"action_id": 6, "x": 56, "y": 28}
BLUE4_ANVIL_CLICK = {"action_id": 6, "x": 51, "y": 41}
# L5 anvil-button clicks (buffer steps 379/383/384): blue blob (comp 47),
# blue red-glyph mover (comp 16), yellow navy-bar slider.
BLUE5_BLOB_CLICK = {"action_id": 6, "x": 51, "y": 41}
BLUE5_GLYPH_CLICK = {"action_id": 6, "x": 50, "y": 29}
YELLOW5_ANVIL_CLICK = {"action_id": 6, "x": 47, "y": 22}
MAG5_ANVIL_CLICK = {"action_id": 6, "x": 52, "y": 48}


def _planner_actions(frame, available_actions):
    """Action set for the search, level-aware so the same BFS solves either
    layout: the moves plus whichever right-panel toggle clicks exist."""
    moves = [a for a in (A_UP, A_DOWN, A_LEFT, A_RIGHT) if a in available_actions]
    actions = list(moves)
    if 6 in available_actions:
        lvl = _detect_level(frame)
        if lvl == 5:
            actions += [BLUE5_BLOB_CLICK, BLUE5_GLYPH_CLICK, YELLOW5_ANVIL_CLICK,
                        MAG5_ANVIL_CLICK]
        elif lvl == 4:
            actions += [YELLOW4_ANVIL_CLICK, MAG4_ANVIL_CLICK, GRAY4_ANVIL_CLICK,
                        BLUE4_ANVIL_CLICK]
        elif lvl == 3:
            actions += [RED3_ANVIL_CLICK, BLUE3_ANVIL_CLICK, MAG3_ANVIL_CLICK,
                        WHITE3_ANVIL_CLICK]
        elif lvl == 2:
            actions += [PINK_ANVIL_CLICK, BLUE2_ANVIL_CLICK, RED2_ANVIL_CLICK]
        else:
            actions += [BLUE_PREVIEW_CLICK, RED_PREVIEW_CLICK]
    return actions


def _planner_key(g):
    """Abstract dedup key: green position plus the level's toggle-state bits.
    Reads only the regions that actually exist in the detected level so the
    search does not collapse distinct L2 states onto L1 sentinels."""
    p = _find_player(g)
    gp = (p[0], p[1]) if p else None
    lvl = _detect_level(g)
    if lvl == 4:
        # Navy-bar positions (leftmost navy col per row band) + token phase +
        # gray-toggle form. Captures the slide/flip/couple state the anvils set.
        navy = tuple(sorted((r, c) for r, row in enumerate(g)
                            for c, v in enumerate(row) if v == NAVY))
        blues = tuple(_region_full(g, reg, BLUE) for reg in BLUE4_REGIONS)
        return (gp, navy, g[24][4], g[40][16], g[20][8], blues)
    if lvl == 3:
        blues = tuple(_region_full(g, reg, BLUE) for reg in BLUE3_REGIONS)
        mag = g[16][6]  # magenta token-pair phase (one cell is enough)
        white_solid = _region_full(g, WHITE3_REGIONS[0], WHITE)
        return (gp, _red2_in_v(g), blues, mag, white_solid)
    if lvl == 2:
        return (gp, _pink_in_v(g), _red2_in_v(g),
                _has_token(g, RED_TOKEN, RED),
                _region_full(g, BLUE2_BLOCK, BLUE))
    if lvl == 5:
        # green pos + navy-bar leftmost col + red-glyph top row + blue blobs.
        navy = [c for r in range(len(g)) for c in range(RIGHT_X)
                if g[r][c] == NAVY]
        red = [r for r in range(len(g)) for c in range(RIGHT_X)
               if g[r][c] == RED]
        blues = tuple(_region_full(g, reg, BLUE) for reg in BLUE4_REGIONS)
        return (gp, min(navy) if navy else -1, min(red) if red else -1, blues)
    return (gp, _bar_is_v(g, RED, (RED_H,), (RED_V,)), _blob_solid(g),
            _region_count(g, BLUE_A, BLUE) >= 8)


def planner(frame, available_actions=None, max_depth=None):
    """Budgeted goal-directed BFS over an abstract state (green-token position +
    the level's toggle-state bits). The reachable abstract space is small (green
    over a few dozen floor cells x a handful of toggle bits), so BFS terminates
    fast; we prune by the abstract key and return the first reward-reaching
    action list, else None."""
    if available_actions is None:
        available_actions = [0, 1, 2, 3, 4, 6, 7]
    if max_depth is None:
        max_depth = 40

    start = [list(row) for row in frame]
    if reward_function(start, None, start)[0] > 0:
        return []
    actions = _planner_actions(start, available_actions)

    from collections import deque

    seen = {_planner_key(start)}
    q = deque([(start, [])])
    while q:
        g, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for a in actions:
            ng = transition_function(g, a)
            if reward_function(g, a, ng)[0] > 0:
                return path + [a]
            k = _planner_key(ng)
            if k in seen:
                continue
            seen.add(k)
            q.append((ng, path + [a]))
    return None


def move_counter_mask():
    """Row 63 is the HUD move-counter strip (ticks roughly every other move;
    not predictable from a single frame). Exclude it from verification."""
    return [(63, c) for c in range(64)]


def extract_objects(frame):
    """Lightweight visual-object abstraction reusing the transition ontology.

    Connected components over non-background cells, with semantic typing for
    the families this model recognises (player token, framed tokens, right-
    panel target shapes)."""
    grid = [[int(v) for v in row] for row in frame]
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    # Treat the two panel backgrounds (most common colours) as background.
    bg = set()
    for v, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:2]:
        bg.add(v)
    bg.add(3)  # outer border / HUD dark

    def type_for(color, x, y):
        right = x >= 34
        if color == GREEN:
            return "player"          # green cursor token
        if color == YELLOW:
            # Left-board yellow is the delivery TARGET; a yellow shape in the
            # RIGHT panel (L4) is a clickable anvil CONTROL, not the goal.
            return "control_yellow" if right else "target"
        if color == FRAME:
            return "frame"
        if right and color == RED:
            return "control_red"     # clickable toggle control (red preview)
        if right and color == BLUE:
            return "control_blue"    # clickable toggle control (blue preview)
        if color == RED:
            return "piece_red"       # toggleable red workspace piece
        if color == BLUE:
            return "piece_blue"      # toggleable blue workspace piece(s)
        if color == DARKRED:
            return "base_darkred"    # fixed base under red V-position
        return f"object_{color}"

    seen = [[False] * w for _ in range(h)]
    objects = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0] or grid[y0][x0] in bg:
                continue
            color = grid[y0][x0]
            stack = [(x0, y0)]
            seen[y0][x0] = True
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] \
                            and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            x, y = min(xs), min(ys)
            bw, bh = max(xs) - x + 1, max(ys) - y + 1
            pixels = [[-1] * bw for _ in range(bh)]
            for cx, cy in cells:
                pixels[cy - y][cx - x] = color
            objects.append({
                "name": f"cc_{color}_{len(objects):03d}",
                "type": type_for(color, x, y),
                "tags": [f"color_{color}"],
                "x": x, "y": y, "w": bw, "h": bh,
                "pixels": pixels, "layer": 0, "visible": True,
            })
    return objects
