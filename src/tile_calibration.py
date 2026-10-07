"""
tile_calibration.py
-------------------
Robust floor-tile calibration meant to be imported (e.g. from detect-motion.py)
and run ONCE on the first frame.

    from tile_calibration import calibrate_from_frame
    result = calibrate_from_frame(first_frame)
    if result is None:
        ...  # not enough clean grid visible -> try the next frame / keep fallback
    pts_image, pts_real = result.pts_image, result.pts_real

What it adds on top of tile_camera_calibration.py
  1. No photo-specific masks.  Non-floor clutter is rejected by geometry:
       - colourful objects (saturation) are removed from the edge map
       - grid lines must converge to a common vanishing point (RANSAC)
       - line spacing must follow the projective spacing of equal 60 cm tiles;
         spurious lines are dropped, missing lines are bridged (index skips)
       - an intersection is only kept if BOTH lines have visible segments near it
  2. A quality gate: returns None instead of silently returning bad points
     (needs >= min_points and a small homography reprojection error in cm).
"""

from dataclasses import dataclass, field
from itertools import combinations, product
import cv2
import numpy as np



# --------------------------------------------------------------------------
# Small geometry helpers (copied so this module has no other file dependency)
# --------------------------------------------------------------------------
def ordinal(n: int) -> str:
    """Converts an integer to its ordinal string (e.g., 1 -> '1st', 2 -> '2nd')."""
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def line_intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    """
    Computes the 2D intersection point (x, y) of two lines defined by (A, B, C) 
    where Ax + By + C = 0. Returns None if lines are parallel.
    """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return None
    x = (B1 * C2 - B2 * C1) / det
    y = (A2 * C1 - A1 * C2) / det
    return np.array([x, y], dtype=np.float32)


def fit_line_from_points(pts: list) -> np.ndarray:
    """
    Fits a normalized line Ax + By + C = 0 to a set of 2D points using 
    Total Least Squares (cv2.fitLine).
    """
    pts_arr = np.array(pts, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    A = -vy
    B = vx
    C = vy * x0 - vx * y0
    norm = np.sqrt(A**2 + B**2)
    return np.array([A / norm, B / norm, C / norm])


def cluster_by_gap(items: list, max_gap: float = 25.0) -> list:
    """
    Clusters 1D items (sorted by position) if the gap between consecutive items 
    is less than max_gap.
    """
    if not items:
        return []
    sorted_items = sorted(items, key=lambda x: x[0])
    clusters = [[sorted_items[0]]]
    for item in sorted_items[1:]:
        if item[0] - clusters[-1][-1][0] <= max_gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


# --------------------------------------------------------------------------
# Line-family filtering
# --------------------------------------------------------------------------
@dataclass
class GridLine:
    pos: float            # position along the reference transversal (px)
    eq: np.ndarray        # normalised (A, B, C), Ax+By+C=0
    mid: np.ndarray       # a point on the line near the image centre
    extent: tuple         # (s_min, s_max) of observed segments along the line
    origin: np.ndarray    # reference point for `extent`
    direction: np.ndarray # unit direction
    idx: int = -1         # integer tile index assigned later (0-based)


def _vp_filter(lines, tol_deg=1.5):
    """Keep the largest set of lines that (nearly) share one vanishing point."""
    if len(lines) < 4:
        return lines

    def resid(line, vp):
        u = vp - line.mid
        n = np.linalg.norm(u)
        if n < 1e-6:
            return 0.0
        c = abs(np.dot(line.direction, u / n))
        return np.degrees(np.arccos(min(1.0, c)))

    best, best_score = lines, (-1, 0.0)
    for a, b in combinations(lines, 2):
        vp = line_intersection(a.eq, b.eq)
        if vp is None:
            # exactly parallel: use their direction as the "VP at infinity"
            vp = a.mid + a.direction * 1e7
        res = [resid(l, vp) for l in lines]
        inl = [l for l, r in zip(lines, res) if r <= tol_deg]
        score = (len(inl), -sum(r for r in res if r <= tol_deg))
        if score > best_score:
            best, best_score = inl, score
    return best


SKIP_PENALTY_PX = 6.0   # a skipped tile must buy >6 px of RMS improvement


def _fit_spacing(ts, max_skip=3):
    """
    ts: sorted positions of lines along a transversal.  Equal world spacing
    maps to t = (a*k + b) / (c*k + 1) for integer k.  Brute-force the integer
    gaps (1..max_skip tiles between consecutive lines), return the assignment
    with the lowest RMS residual (small penalty per skipped tile).
    """
    n = len(ts)
    if n < 3:
        return list(range(n)), 0.0, 0.0
    ts = np.asarray(ts, float)
    best = (None, np.inf, np.inf)  # ks, rms, score
    for gaps in product(range(1, max_skip + 1), repeat=n - 1):
        k = np.concatenate([[0], np.cumsum(gaps)]).astype(float)
        # t*(c*k+1) = a*k + b   ->   t = a*k + b - c*k*t
        M = np.column_stack([k, np.ones(n), -k * ts])
        sol, *_ = np.linalg.lstsq(M, ts, rcond=None)
        a, b, c = sol
        den = c * k + 1
        if np.any(den <= 1e-6):
            continue
        pred = (a * k + b) / den
        rms = float(np.sqrt(np.mean((pred - ts) ** 2)))
        score = rms + SKIP_PENALTY_PX * (sum(gaps) - (n - 1))
        if score < best[2]:
            best = (k.astype(int).tolist(), rms, score)
    if best[0] is None:
        return list(range(n)), np.inf, np.inf
    return best[0], best[1], best[2]


def _spacing_filter(lines, rms_tol=3.0, max_drop=3):
    """Drop lines that break the equal-tile-spacing pattern, assign indices."""
    lines = sorted(lines, key=lambda l: l.pos)
    ks, rms, score = _fit_spacing([l.pos for l in lines])
    drops = 0
    while rms > rms_tol and len(lines) > 4 and drops < max_drop:
        trials = []
        for i in range(len(lines)):
            sub = lines[:i] + lines[i + 1:]
            k2, r2, sc2 = _fit_spacing([l.pos for l in sub])
            trials.append((sc2, r2, i, k2))
        sc2, r2, i, k2 = min(trials, key=lambda t: t[0])
        if sc2 >= score * 0.8:      # removal doesn't explain the data better
            break
        lines.pop(i)
        ks, rms, score = k2, r2, sc2
        drops += 1
    for l, k in zip(lines, ks):
        l.idx = int(k)
    return lines, rms


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _make_grid_lines(clusters, ref_x, ref_y, vertical):
    out = []
    for cl in clusters:
        pts = [p for item in cl for p in (item[1], item[2])]
        A, B, C = fit_line_from_points(pts)
        if vertical:
            if abs(A) < 1e-9:
                continue
            pos = -(B * ref_y + C) / A
            mid = np.array([pos, ref_y])
        else:
            if abs(B) < 1e-9:
                continue
            pos = -(A * ref_x + C) / B
            mid = np.array([ref_x, pos])
        d = np.array([B, -A])
        d /= np.linalg.norm(d)
        s = [np.dot(np.asarray(p, float) - mid, d) for p in pts]
        out.append(GridLine(pos, np.array([A, B, C]), mid, (min(s), max(s)), mid, d))
    return out


def _supported(line, pt, margin):
    s = np.dot(pt - line.origin, line.direction)
    return line.extent[0] - margin <= s <= line.extent[1] + margin


@dataclass
class Calibration:
    pts_image: np.ndarray           # (N,2) float32, pixels in the frame given
    pts_real: np.ndarray            # (N,2) float32, cm
    H: np.ndarray                   # image -> real (cm)
    rms_cm: float
    corners: list = field(default_factory=list)
    debug_frame: np.ndarray = None


LAST_FAILURE = ""


def _fail(msg):
    global LAST_FAILURE
    LAST_FAILURE = msg
    return None


def calibrate_from_frame(frame, tile_size_cm=60.0, min_points=6,
                         max_rms_cm=2.0, support_margin=30.0, min_lines=4,
                         sat_thresh=70, draw_debug=True, ignore_top_frac=0.2):
    """Return a Calibration or None if the grid could not be found reliably."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- edge map: dark thin lines on a bright, near-grey floor -----------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    edges = cv2.Canny(cv2.GaussianBlur(tophat, (5, 5), 0), 30, 120)
    # coloured objects (stool, cables, boxes) are not floor: drop their edges
    colourful = cv2.dilate(
        ((hsv[..., 1] > sat_thresh) & (hsv[..., 2] > 50)).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
    edges[colourful > 0] = 0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=80, maxLineGap=30)
    if lines is None:
        _fail("no hough lines")

    ref_x, ref_y = w / 2.0, h / 2.0
    v_items, h_items = [], []
    for ln in lines:
        x1, y1, x2, y2 = ln.flatten()
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        p1, p2 = np.array([x1, y1]), np.array([x2, y2])
        if 65 <= ang <= 115:
            xm = x1 + (ref_y - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
            v_items.append((xm, p1, p2))
        elif ang <= 30 or ang >= 150:
            ym = y1 + (ref_x - x1) * (y2 - y1) / (x2 - x1) if x2 != x1 else y1
            h_items.append((ym, p1, p2))

    v_lines = _make_grid_lines(cluster_by_gap(v_items, 25.0), ref_x, ref_y, True)
    h_lines = _make_grid_lines(cluster_by_gap(h_items, 25.0), ref_x, ref_y, False)
    if len(v_lines) < 2 or len(h_lines) < 2:
        _fail(f"too few raw lines v={len(v_lines)} h={len(h_lines)}")

    v_lines, _ = _spacing_filter(_vp_filter(v_lines))
    h_lines, _ = _spacing_filter(_vp_filter(h_lines))
    # with <4 lines a missing/extra line cannot be told apart from a real gap
    if len(v_lines) < min_lines or len(h_lines) < min_lines:
        _fail(f"too few lines after filter v={len(v_lines)} h={len(h_lines)}")

    # image y grows downward, so the largest-y line is the bottom one (row 1)
    max_row = max(l.idx for l in h_lines)
    for l in h_lines:
        l.idx = max_row - l.idx

    # --- intersections, only where both lines are actually visible ---------
    found = []
    for vl in v_lines:
        for hl in h_lines:
            pt = line_intersection(vl.eq, hl.eq)
            if pt is None or not (0 <= pt[0] < w and 0 <= pt[1] < h):
                continue
            if pt[1] < ignore_top_frac * h:
                continue
            if _supported(vl, pt, support_margin) and _supported(hl, pt, support_margin):
                found.append((vl.idx, hl.idx, pt))
    if len(found) < min_points:
        _fail(f"too few supported intersections ({len(found)})")

    pts = np.array([f[2] for f in found], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pts = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (7, 7), (-1, -1), crit).reshape(-1, 2)
    real = np.array([[f[0] * tile_size_cm, f[1] * tile_size_cm] for f in found], np.float32)

    H, mask = cv2.findHomography(pts, real, cv2.RANSAC, 5.0)  # 5 cm tolerance
    if H is None:
        _fail("homography failed")
    keep = mask.ravel().astype(bool)
    if keep.sum() < min_points:
        _fail("too few ransac inliers")
    # refit on inliers, then measure error in cm
    H, _ = cv2.findHomography(pts[keep], real[keep], 0)
    proj = cv2.perspectiveTransform(pts[keep].reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - real[keep]) ** 2, axis=1))))
    if rms > max_rms_cm:
        _fail(f"high reprojection error {rms:.1f} cm")

    pts, real = pts[keep], real[keep]
    kept_found = [f for f, k in zip(found, keep) if k]
    corners = [dict(id=f"P{i+1:02d}", col=f[0] + 1, row=f[1] + 1,
                    label=f"({f[0]+1}L, {f[1]+1}B)") for i, f in enumerate(kept_found)]

    dbg = None
    if draw_debug:
        dbg = frame.copy()
        for l in v_lines:
            p0 = (int(-l.eq[2] / l.eq[0]), 0)
            p1 = (int(-(l.eq[1] * h + l.eq[2]) / l.eq[0]), h)
            cv2.line(dbg, p0, p1, (0, 255, 255), 1)
        for l in h_lines:
            p0 = (0, int(-l.eq[2] / l.eq[1]))
            p1 = (w, int(-(l.eq[0] * w + l.eq[2]) / l.eq[1]))
            cv2.line(dbg, p0, p1, (255, 255, 0), 1)
        for c, p in zip(corners, pts):
            q = (int(round(p[0])), int(round(p[1])))
            cv2.circle(dbg, q, 5, (0, 255, 0), 2)
            cv2.putText(dbg, c["label"], (q[0] + 6, q[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return Calibration(pts.astype(np.float32), real.astype(np.float32), H, rms, corners, dbg)


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "floor_cropped.jpg")
    res = calibrate_from_frame(img)
    if res is None:
        print("Calibration failed (grid not reliably found)")
    else:
        print(f"{len(res.pts_image)} points, reprojection RMS {res.rms_cm:.2f} cm")
        print("pts_image =", np.round(res.pts_image, 2).tolist())
        print("pts_real  =", res.pts_real.tolist())
        cv2.imwrite("calib_debug.png", res.debug_frame)
