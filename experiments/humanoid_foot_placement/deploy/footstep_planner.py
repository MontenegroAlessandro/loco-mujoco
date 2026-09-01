from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math

import numpy as np


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def hypot2(x: float, y: float) -> float:
    return math.hypot(x, y)


from dataclasses import dataclass

@dataclass
class StepConstraints:
    r_max: float = 0.40
    r_min: float= 0.2
    feet_distance: float = 0.15
    dyaw_max: float = math.radians(30)


def relative_step_local(u: State, v: State) -> Tuple[float, float, float]:
    """Return (dx, dy, dyaw) in u's local frame, with dy mirrored to match our convention."""
    dxw = v.x - u.x
    dyw = v.y - u.y
    c = math.cos(u.yaw)
    s = math.sin(u.yaw)

    # world -> local
    dx =  c * dxw + s * dyw
    dy = -s * dxw + c * dyw

    dyaw = wrap_to_pi(v.yaw - u.yaw)
    return dx, dy, dyaw


def feasible_transition(u: State, v: State, cons: StepConstraints) -> bool:
    dx, dy, dyaw = relative_step_local(u, v)

    if np.sqrt(dx ** 2 + dy ** 2) < cons.r_min:
        return False

    if np.sqrt(dx ** 2 + dy ** 2) > cons.r_max:
        return False

    sign = +1 if u.foot == 1 else -1
    if sign * dy < cons.feet_distance:
        return False

    if abs(dyaw) > cons.dyaw_max:
        return False

    return True



# lattice config
@dataclass(frozen=True)
class LatticeConfig:
    xy_res: float = 0.01          # meters
    theta_bins: int = 72          # discretization of yaw
    step_cost: float = 0.05       # cost per step
    yaw_cost: float = 0.05        # yaw penalty weight

    @property
    def theta_res(self) -> float:
        return 2.0 * math.pi / float(self.theta_bins)

# -------------------------
# state
# -------------------------
@dataclass(frozen=True)
class State:
    x: float
    y: float
    yaw: float     # radians, wrapped to [-pi, pi)
    foot: int      # 0 = left support, 1 = right support

    def key(self, cfg: LatticeConfig) -> Tuple[int, int, int, int]:
        """Discrete key for hashing / closed set."""
        ix = int(round(self.x / cfg.xy_res))
        iy = int(round(self.y / cfg.xy_res))
        it = int(round(wrap_to_pi(self.yaw) / cfg.theta_res)) % cfg.theta_bins
        return (ix, iy, it, int(self.foot))

def snap_stance(s: State, cfg: LatticeConfig) -> State:
    """Snap continuous stance to lattice."""
    x = round(s.x / cfg.xy_res) * cfg.xy_res
    y = round(s.y / cfg.xy_res) * cfg.xy_res
    yaw = round(wrap_to_pi(s.yaw) / cfg.theta_res) * cfg.theta_res
    yaw = wrap_to_pi(yaw)
    return State(x=x, y=y, yaw=yaw, foot=s.foot)



@dataclass(frozen=True)
class Action:
    dx: float
    dy: float
    dyaw: float

def action_set(cfg: LatticeConfig) -> List[Action]:
    DX_FWD = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    DY_NORM = [0.18, 0.20, 0.25, 0.30, 0.40, 0.45]
    DYAW = [0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5, +6, -6]

    actions: List[Action] = []
    for dx_mag in DX_FWD:
        for sign in (+1, -1):
            dx = sign * dx_mag
            for dy in DY_NORM:
                for k in DYAW:
                    actions.append(Action(dx=dx, dy=dy, dyaw=k * cfg.theta_res))

    uniq = {}
    for a in actions:
        key = (int(round(a.dx / cfg.xy_res)),
               int(round(a.dy / cfg.xy_res)),
               int(round(wrap_to_pi(a.dyaw) / cfg.theta_res)))
        uniq[key] = a
    return list(uniq.values())



def apply_action(s: State, a: Action) -> State:
    """
    Apply action in support-foot local frame.
    Mirror dy for left vs right support to keep symmetry.
    """
    # mirror lateral direction depending on which foot is support:
    # convention: foot=0 (L support) => swing is R => dy as-is
    #             foot=1 (R support) => swing is L => mirror dy
    dy = -a.dy if s.foot == 0 else a.dy

    cy = math.cos(s.yaw)
    sy = math.sin(s.yaw)

    wx = cy * a.dx - sy * dy
    wy = sy * a.dx + cy * dy

    x2 = s.x + wx
    y2 = s.y + wy
    yaw2 = wrap_to_pi(s.yaw + a.dyaw)

    # alternate foot each step
    foot2 = 1 - s.foot
    return State(x=x2, y=y2, yaw=yaw2, foot=foot2)


def step_cost(s: State, s2: State, cfg: LatticeConfig) -> float:
    d = hypot2(s2.x - s.x, s2.y - s.y)
    dyaw = abs(wrap_to_pi(s2.yaw - s.yaw))
    return d + cfg.step_cost + cfg.yaw_cost * dyaw



# -------------------------
# Broad-phase: Grid Hash
# -------------------------
class GridHash:
    """Spatial hash for obstacle centers (fast local query)."""
    def __init__(self, obstacle_centers: List[Tuple[float, float]], cell_size: float):
        self.cs = float(cell_size)
        self.grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        for (ox, oy) in obstacle_centers:
            key = self._key(ox, oy)
            self.grid.setdefault(key, []).append((ox, oy))

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.cs)), int(math.floor(y / self.cs)))

    def query(self, x: float, y: float, radius: float) -> List[Tuple[float, float]]:
        """Return obstacle centers in a neighborhood around (x,y)."""
        r = float(radius)
        ix0, iy0 = self._key(x - r, y - r)
        ix1, iy1 = self._key(x + r, y + r)
        out: List[Tuple[float, float]] = []
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                out.extend(self.grid.get((ix, iy), []))
        return out


# -------------------------
# Narrow-phase: SAT
# Oriented rectangle (foot) vs axis-aligned square (obstacle)
# -------------------------
def _dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by

def _project_points_on_axis(points: List[Tuple[float, float]], ax: float, ay: float) -> Tuple[float, float]:
    mn = float("inf")
    mx = -float("inf")
    for (px, py) in points:
        v = _dot(px, py, ax, ay)
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    return mn, mx

def _interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])

def _rect_corners_world(cx: float, cy: float, yaw: float, hx: float, hy: float) -> List[Tuple[float, float]]:
    """Corners of oriented rectangle centered at (cx,cy) with half-sizes hx,hy."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    # local corners
    local = [(+hx, +hy), (+hx, -hy), (-hx, -hy), (-hx, +hy)]
    out = []
    for (lx, ly) in local:
        wx = cx + c * lx - s * ly
        wy = cy + s * lx + c * ly
        out.append((wx, wy))
    return out

def _aabb_square_corners(ox: float, oy: float, half: float) -> List[Tuple[float, float]]:
    return [
        (ox + half, oy + half),
        (ox + half, oy - half),
        (ox - half, oy - half),
        (ox - half, oy + half),
    ]

def sat_oriented_rect_vs_aabb_square(
    foot_cx: float, foot_cy: float, foot_yaw: float, foot_hx: float, foot_hy: float,
    obs_cx: float, obs_cy: float, obs_half: float,
) -> bool:
    """
    Return True if intersecting (collision).
    SAT axes to test:
      - foot local x-axis, foot local y-axis
      - world x-axis, world y-axis (AABB axes)
    """
    foot_pts = _rect_corners_world(foot_cx, foot_cy, foot_yaw, foot_hx, foot_hy)
    obs_pts  = _aabb_square_corners(obs_cx, obs_cy, obs_half)

    # axes: foot axes
    fx = math.cos(foot_yaw)
    fy = math.sin(foot_yaw)
    axes = [
        (fx, fy),          # foot x-axis
        (-fy, fx),         # foot y-axis
        (1.0, 0.0),        # world x
        (0.0, 1.0),        # world y
    ]

    for (ax, ay) in axes:
        f_int = _project_points_on_axis(foot_pts, ax, ay)
        o_int = _project_points_on_axis(obs_pts,  ax, ay)
        if not _interval_overlap(f_int, o_int):
            return False  # separated -> no collision
    return True  # overlap on all axes -> collision


# -------------------------
# Collision checker for square obstacles + rectangular feet
# -------------------------

def _segment_intersects_aabb(x0, y0, x1, y1, minx, maxx, miny, maxy) -> bool:
    """Segment [p0,p1] intersects axis-aligned box."""
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0

    def clip(p, q, t0, t1):
        if abs(p) < 1e-12:
            if q < 0.0:
                return False, t0, t1
            return True, t0, t1
        r = q / p
        if p < 0.0:
            if r > t1: return False, t0, t1
            if r > t0: t0 = r
        else:
            if r < t0: return False, t0, t1
            if r < t1: t1 = r
        return True, t0, t1

    ok, t0, t1 = clip(-dx, x0 - minx, t0, t1)
    if not ok: return False
    ok, t0, t1 = clip( dx, maxx - x0, t0, t1)
    if not ok: return False
    ok, t0, t1 = clip(-dy, y0 - miny, t0, t1)
    if not ok: return False
    ok, t0, t1 = clip( dy, maxy - y0, t0, t1)
    if not ok: return False
    return True


def _segment_hits_square_inflated(
    x0: float, y0: float, x1: float, y1: float,
    ox: float, oy: float, half: float,
    inflate: float,
) -> bool:
    """Capsule approx via inflating the square by inflate and testing segment vs AABB."""
    h = half + inflate
    minx, maxx = ox - h, ox + h
    miny, maxy = oy - h, oy + h
    return _segment_intersects_aabb(x0, y0, x1, y1, minx, maxx, miny, maxy)



@dataclass
class FootRect:
    length: float = 0.22  # along foot forward (x in foot frame)
    width: float = 0.10   # lateral (y in foot frame)


def _abs_dot(ax: float, ay: float, bx: float, by: float) -> float:
    return abs(ax * bx + ay * by)

def obb_overlap_2d(
    a_cx: float, a_cy: float, a_yaw: float, a_hx: float, a_hy: float,
    b_cx: float, b_cy: float, b_yaw: float, b_hx: float, b_hy: float,
) -> bool:
    """
    True if two OBBs intersect.
    Tests 4 separating axes: A's x/y and B's x/y.
    """
    # A axes
    cA, sA = math.cos(a_yaw), math.sin(a_yaw)
    a_xx, a_xy = cA, sA
    a_yx, a_yy = -sA, cA

    # B axes
    cB, sB = math.cos(b_yaw), math.sin(b_yaw)
    b_xx, b_xy = cB, sB
    b_yx, b_yy = -sB, cB

    # center delta
    dx = b_cx - a_cx
    dy = b_cy - a_cy

    # test axes
    axes = [(a_xx, a_xy), (a_yx, a_yy), (b_xx, b_xy), (b_yx, b_yy)]

    for ax, ay in axes:
        # distance between centers projected onto axis
        dist = abs(dx * ax + dy * ay)

        # projection radius of A onto axis
        rA = a_hx * _abs_dot(a_xx, a_xy, ax, ay) + a_hy * _abs_dot(a_yx, a_yy, ax, ay)
        # projection radius of B onto axis
        rB = b_hx * _abs_dot(b_xx, b_xy, ax, ay) + b_hy * _abs_dot(b_yx, b_yy, ax, ay)

        if dist > rA + rB:
            return False  # separated
    return True  # overlap on all axes

def _segment_intersects_aabb(x0, y0, x1, y1, minx, maxx, miny, maxy) -> bool:
    """Segment [p0,p1] intersects axis-aligned box."""
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0

    def clip(p, q, t0, t1):
        if abs(p) < 1e-12:
            if q < 0.0:
                return False, t0, t1
            return True, t0, t1
        r = q / p
        if p < 0.0:
            if r > t1: return False, t0, t1
            if r > t0: t0 = r
        else:
            if r < t0: return False, t0, t1
            if r < t1: t1 = r
        return True, t0, t1

    ok, t0, t1 = clip(-dx, x0 - minx, t0, t1)
    if not ok: return False
    ok, t0, t1 = clip( dx, maxx - x0, t0, t1)
    if not ok: return False
    ok, t0, t1 = clip(-dy, y0 - miny, t0, t1)
    if not ok: return False
    ok, t0, t1 = clip( dy, maxy - y0, t0, t1)
    if not ok: return False
    return True

def segment_hits_oriented_square_inflated(
    x0: float, y0: float,
    x1: float, y1: float,
    cone: TrafficCone,
    inflate: float,
) -> bool:
    """
    Check whether segment [p0,p1] intersects an oriented square (cone) inflated by `inflate`.
    Implementation: transform segment into cone-local frame => AABB segment test.
    """
    # translate to cone center
    dx0, dy0 = x0 - cone.cx, y0 - cone.cy
    dx1, dy1 = x1 - cone.cx, y1 - cone.cy

    # rotate by -yaw (world -> cone local)
    c = math.cos(cone.yaw)
    s = math.sin(cone.yaw)

    x0l =  c * dx0 + s * dy0
    y0l = -s * dx0 + c * dy0
    x1l =  c * dx1 + s * dy1
    y1l = -s * dx1 + c * dy1

    h = cone.half + float(inflate)
    return _segment_intersects_aabb(x0l, y0l, x1l, y1l, -h, +h, -h, +h)



@dataclass(frozen=True)
class TrafficCone:
    cx: float
    cy: float
    yaw: float
    half: float

class CollisionChecker:
    def __init__(
        self,
        cones: List[TrafficCone],
        foot: FootRect = FootRect(),
        swing_radius: float = 0.06,
        corridor_radius: float = 0.12,
        margin: float = 0.01,
        grid_cell_size: float = 0.60,
    ):
        """
        cones: list of TrafficCone (oriented squares in world XY)
        foot: rectangle approximation for the foot sole (OBB)
        margin: safety margin added to both foot and obstacles
        """
        self.cones = list(cones)

        self.foot_hx = 0.5 * float(foot.length)
        self.foot_hy = 0.5 * float(foot.width)
        self.margin = float(margin)

        self.swing_radius = float(swing_radius)
        self.corridor_radius = float(corridor_radius)

        # Broadphase: store only centers in grid
        self._centers: List[Tuple[float, float]] = [(c.cx, c.cy) for c in self.cones]
        self.grid = GridHash(self._centers, cell_size=grid_cell_size)

        # Map center -> cone (assumes unique centers; if not unique, use indices in GridHash)
        self._cone_by_center: Dict[Tuple[float, float], TrafficCone] = {
            (c.cx, c.cy): c for c in self.cones
        }

        # Precompute safe global query radius for footstep check:
        max_half = max((c.half for c in self.cones), default=0.0)
        obs_r = math.sqrt(2.0) * (max_half + self.margin)  # circumradius of (inflated) square
        foot_r = math.sqrt((self.foot_hx + self.margin) ** 2 + (self.foot_hy + self.margin) ** 2)
        self.query_radius_footstep = obs_r + foot_r

    def _get_cone(self, cx: float, cy: float) -> Optional[TrafficCone]:
        return self._cone_by_center.get((cx, cy), None)

    def is_footstep_valid(self, s: "State") -> bool:
        """
        Foot placement collision: foot OBB vs cone OBB squares.
        """
        fhx = self.foot_hx + self.margin
        fhy = self.foot_hy + self.margin

        candidates = self.grid.query(s.x, s.y, self.query_radius_footstep)
        for (cx, cy) in candidates:
            cone = self._get_cone(cx, cy)
            if cone is None:
                continue

            oh = cone.half + self.margin  # inflate obstacle too
            if obb_overlap_2d(
                a_cx=s.x, a_cy=s.y, a_yaw=s.yaw, a_hx=fhx, a_hy=fhy,
                b_cx=cone.cx, b_cy=cone.cy, b_yaw=cone.yaw, b_hx=oh, b_hy=oh,
            ):
                return False
        return True

    def is_transition_valid(self, u: State, v: State) -> bool:
        if self._segment_hits_any_cone(u.x, u.y, v.x, v.y, inflate=self.corridor_radius + self.margin):
            return False

        return True

    def _segment_hits_any_cone(self, x0, y0, x1, y1, inflate: float) -> bool:
        mx = 0.5 * (x0 + x1)
        my = 0.5 * (y0 + y1)
        seg_len = math.hypot(x1 - x0, y1 - y0)
        max_half = max((c.half for c in self.cones), default=0.0)
        query_r = 0.5 * seg_len + math.sqrt(2.0) * (max_half + inflate)

        for (cx, cy) in self.grid.query(mx, my, query_r):
            cone = self._get_cone(cx, cy)
            if cone is None:
                continue
            if segment_hits_oriented_square_inflated(x0, y0, x1, y1, cone=cone, inflate=inflate):
                return True
        return False


def successors(s: State, actions: List[Action], cfg: LatticeConfig, cc: CollisionChecker, cons: StepConstraints):
    out = []
    for a in actions:
        s2_cont = apply_action(s, a)

        if not cc.is_footstep_valid(s2_cont):
            continue

        if not cc.is_transition_valid(s, s2_cont):
            continue

        if not feasible_transition(s, s2_cont, cons):
            continue

        s2 = snap_stance(s2_cont, cfg)
        c = step_cost(s, s2, cfg)
        out.append((s2, c))
    return out




@dataclass(frozen=True)
class Goal:
    gx: float
    gy: float
    gyaw: float = 0.0
    pos_tol: float = 0.20      # meters
    yaw_tol: float = math.radians(20)  # rad
    require_yaw: bool = False  # 先默认 False，更稳

def is_goal(s: State, goal: Goal) -> bool:
    if hypot2(s.x - goal.gx, s.y - goal.gy) > goal.pos_tol:
        return False
    if goal.require_yaw:
        if abs(wrap_to_pi(s.yaw - goal.gyaw)) > goal.yaw_tol:
            return False
    return True

def h_euclid(s: State, goal: Goal) -> float:
    return hypot2(goal.gx - s.x, goal.gy - s.y)




def weighted_astar_connect(
    start: State,
    goal: Goal,
    actions: List[Action],
    cfg: LatticeConfig,
    cc: CollisionChecker,
    cons: StepConstraints,
    w: float = 2.0,
    node_limit: int = 3000,
) -> Tuple[bool, List[State], float, int]:
    """
    wA* on the footstep lattice from start to (goal region).
    Designed to be used as R*'s local CONNECT(u,v).

    Notes:
    - Uses discrete keys for closed set and g-values.
    - Stops if node_limit expansions reached.
    """
    start = snap_stance(start, cfg)

    # If start itself invalid, fail early
    if not cc.is_footstep_valid(start):
        return (False, [], float("inf"), 0)

    sk = start.key(cfg)

    # g-score map keyed by discrete state key
    g: Dict[Tuple[int,int,int,int], float] = {sk: 0.0}
    parent: Dict[Tuple[int,int,int,int], Tuple[int,int,int,int]] = {}
    state_of: Dict[Tuple[int,int,int,int], State] = {sk: start}

    # OPEN heap of (f, tie, key)
    # tie-breaker: smaller h preferred
    open_heap: List[Tuple[float, float, Tuple[int,int,int,int]]] = []
    h0 = h_euclid(start, goal)
    heapq.heappush(open_heap, (0.0 + w * h0, h0, sk))

    closed: Set[Tuple[int,int,int,int]] = set()

    expanded = 0
    best_goal_key = None

    while open_heap:
        f, hcur, uk = heapq.heappop(open_heap)
        if uk in closed:
            continue
        u = state_of[uk]
        closed.add(uk)

        # goal check
        if is_goal(u, goal):
            best_goal_key = uk
            break

        expanded += 1
        if expanded >= node_limit:
            break

        # expand successors
        for (v, c_uv) in successors(u, actions, cfg, cc, cons):
            vk = v.key(cfg)
            new_g = g[uk] + c_uv

            old_g = g.get(vk, float("inf"))
            if new_g + 1e-12 < old_g:
                g[vk] = new_g
                parent[vk] = uk
                state_of[vk] = v

                hv = h_euclid(v, goal)
                heapq.heappush(open_heap, (new_g + w * hv, hv, vk))

    if best_goal_key is None:
        return (False, [], float("inf"), expanded)

    # reconstruct path
    path_keys = [best_goal_key]
    while path_keys[-1] != sk:
        path_keys.append(parent[path_keys[-1]])
    path_keys.reverse()

    path = [state_of[k] for k in path_keys]
    cost = g[best_goal_key]
    return (True, path, cost, expanded)



import random
from dataclasses import dataclass

@dataclass
class RStarParams:
    Delta: float = 1.0         # subgoal sampling distance
    k: int = 8                  # number of subgoals per expansion
    goal_bias: float = 0.7      # probability to sample in goal-facing cone
    cone_half_angle: float = math.radians(45)  # cone for goal bias
    w_outer: float = 15.0        # outer weighted A*
    w_local: float = 15.0        # local connector weighted A*
    local_node_limit: int = 500
    max_outer_expansions: int = 250  # budget for outer expansions
    subgoal_pos_tol: float = 0.25    # goal region for CONNECT
    require_yaw_in_connect: bool = False

def angle_of(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)

def sample_subgoals(
    u: State,
    goal_xy: Tuple[float, float],
    cfg: LatticeConfig,
    cc: CollisionChecker,
    params: RStarParams,
    rng: random.Random,
) -> List[State]:
    gx, gy = goal_xy
    dir_to_goal = angle_of(gx - u.x, gy - u.y)

    out: List[State] = []
    attempts = 0
    max_attempts = params.k * 20  # avoid infinite loops

    while len(out) < params.k and attempts < max_attempts:
        attempts += 1

        # sample heading direction
        if rng.random() < params.goal_bias:
            # sample within cone around goal direction
            a = dir_to_goal + rng.uniform(-params.cone_half_angle, params.cone_half_angle)
        else:
            a = rng.uniform(-math.pi, math.pi)

        # sample radius around Delta (small jitter)
        r = params.Delta * rng.uniform(0.85, 1.15)

        vx = u.x + r * math.cos(a)
        vy = u.y + r * math.sin(a)

        # set yaw: face the motion direction (simple + effective)
        vyaw = wrap_to_pi(a)

        # alternate foot is not strictly required for Γ nodes,
        # but it's good to keep "stance state" consistent:
        vfoot = rng.randint(0, 1)

        v = snap_stance(State(vx, vy, vyaw, vfoot), cfg)
        if not cc.is_footstep_valid(v):
            continue

        out.append(v)

    return out


@dataclass
class EdgeResult:
    ok: bool
    cost: float
    path: List[State]
    expanded: int



def outer_h(u: State, goal_xy: Tuple[float, float]) -> float:
    gx, gy = goal_xy
    return hypot2(gx - u.x, gy - u.y)

def connect_uv(
    u: State,
    v: State,
    actions: List[Action],
    cfg: LatticeConfig,
    cc: CollisionChecker,
    cons: StepConstraints,
    params: RStarParams,
) -> EdgeResult:
    goal = Goal(
        gx=v.x, gy=v.y, gyaw=v.yaw,
        pos_tol=params.subgoal_pos_tol,
        yaw_tol=math.radians(25),
        require_yaw=params.require_yaw_in_connect
    )
    ok, path, cost, expanded = weighted_astar_connect(
        u, goal, actions, cfg, cc,
        w=params.w_local,
        node_limit=params.local_node_limit,
        cons=cons
    )
    return EdgeResult(ok=ok, cost=cost, path=path, expanded=expanded)


from typing import Any, Dict, List, Set, Tuple
import heapq
import random
import math

Key = Tuple[int, int, int, int]


def rstar_plan(
        start: State,
        goal_xy: Tuple[float, float],
        actions: List[Action],
        cfg: LatticeConfig,
        cc: CollisionChecker,
        cons: StepConstraints,
        params: RStarParams,
        rng_seed: int = 0,
        goal_region_radius: float = 0.05,
) -> Tuple[bool, List[State], Dict[str, Any]]:
    rng = random.Random(rng_seed)
    w = params.w_outer
    Delta = params.Delta


    s_start = snap_stance(start, cfg)
    if not cc.is_footstep_valid(s_start):
        return (False, [], {"reason": "start_in_collision"})


    Gamma: Dict[Key, State] = {}
    bp: Dict[Key, Key] = {}
    g: Dict[Key, float] = {}
    AVOID: Set[Key] = set()  #


    EXPANDED: Set[Key] = set()
    edges: Dict[Key, List[State]] = {}
    h_cache: Dict[Key, float] = {}

    def get_h(s_key: Key, s_obj: State) -> float:
        if s_key not in h_cache:
            h_cache[s_key] = outer_h(s_obj, goal_xy)
        return h_cache[s_key]

    def add_to_gamma(s_obj: State) -> Key:
        k = s_obj.key(cfg)
        if k not in Gamma: Gamma[k] = s_obj
        return k


    sk_start = add_to_gamma(s_start)
    g[sk_start] = 0.0
    h0 = get_h(sk_start, s_start)

    # Heap: (f, h, key)
    open_heap: List[Tuple[float, float, Key]] = [(w * h0, h0, sk_start)]

    best_goal_key = sk_start
    best_goal_dist = h0
    outer_expanded = 0

    goal_x, goal_y = goal_xy

    reached_goal_region = False
    while open_heap and outer_expanded < params.max_outer_expansions:
        outer_expanded += 1
        print(outer_expanded)
        f_val, h_val, s_key = heapq.heappop(open_heap)


        if s_key in AVOID:
            continue

        s = Gamma[s_key]


        is_verified = (s_key == sk_start) or (s_key in edges)

        if not is_verified:
            parent_key = bp.get(s_key)
            if parent_key is None: continue

            res = connect_uv(Gamma[parent_key], s, actions, cfg, cc, cons, params)

            if not res.ok:
                AVOID.add(s_key)
                continue


            s_act = snap_stance(res.path[-1], cfg)
            s_act_key = add_to_gamma(s_act)

            new_g = g[parent_key] + res.cost


            if new_g > w * math.hypot(s_act.x - s_start.x, s_act.y - s_start.y):
                AVOID.add(s_key)
                AVOID.add(s_act_key)
                continue


            if new_g < g.get(s_act_key, float('inf')):
                g[s_act_key] = new_g
                bp[s_act_key] = parent_key
                edges[s_act_key] = res.path


                h_act = get_h(s_act_key, s_act)
                heapq.heappush(open_heap, (new_g + w * h_act, h_act, s_act_key))


            if s_act_key != s_key:
                AVOID.add(s_key)
            continue


        if s_key in EXPANDED:
            continue
        EXPANDED.add(s_key)

        dist_sq = (s.x - goal_x) ** 2 + (s.y - goal_y) ** 2
        dist = math.sqrt(dist_sq)

        if dist < best_goal_dist:
            best_goal_dist = dist
            best_goal_key = s_key

        if dist <= goal_region_radius:
            reached_goal_region = True
            best_goal_key = s_key
            best_goal_dist = dist
            break


        SUCCS = sample_subgoals(s, goal_xy, cfg, cc, params, rng)


        if dist < Delta:
            SUCCS.append(State(goal_x, goal_y, s.yaw, 1 - s.foot))

        for s_prime in SUCCS:
            sk_prime = add_to_gamma(s_prime)
            if sk_prime in EXPANDED: continue


            g_est = g[s_key] + math.sqrt((s_prime.x - s.x) ** 2 + (s_prime.y - s.y) ** 2)

            if g_est < g.get(sk_prime, float('inf')):
                g[sk_prime] = g_est
                bp[sk_prime] = s_key
                h_prime = get_h(sk_prime, s_prime)
                heapq.heappush(open_heap, (g_est + w * h_prime, h_prime, sk_prime))


    if best_goal_key == sk_start:
        return (True, [s_start], {"reason": "at_start"})

    if not reached_goal_region:
        return (False, [], {
            "reason": "goal_not_reached",
            "best_goal_dist": float(best_goal_dist),
            "expanded": outer_expanded,
            "gamma_size": len(Gamma),
        })

    try:
        path_segments = []
        curr = best_goal_key
        while curr != sk_start:
            seg = edges[curr]
            path_segments.append(seg)
            curr = bp[curr]

        path_segments.reverse()
        stitched = [Gamma[sk_start]]
        for seg in path_segments:
            stitched.extend(seg[1:])

        return (True, stitched, {"expanded": outer_expanded, "gamma_size": len(Gamma)})
    except KeyError:
        return (False, [], {"reason": "reconstruct_failed"})