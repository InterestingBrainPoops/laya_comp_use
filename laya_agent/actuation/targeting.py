"""Where inside an element to click. Pure geometry, no side effects.

The naive centre fails in two ways seen on Windows Terminal:
  - composite controls: the 'New Tab' SplitButton's centre lands on the divider between
    its primary part and its dropdown arrow. Fix: descend into the main child.
  - neighbours: a 'Close Tab' button sits inside its TabItem; the tab's centre can be a few
    pixels from it. Fix: pick the point farthest from every neighbouring box, then nearest
    to the centre among ties.
"""
from __future__ import annotations

import math

from laya_agent.models import Rect, UIElement

MAIN_CHILD_MIN_SHARE = 0.4  # child covering this much of the target is its main part
EDGE_MARGIN = 4  # px kept from the target's own edges
GRID = 7  # candidate points per axis


def contains(outer: Rect, inner: Rect) -> bool:
    return outer.left <= inner.left and outer.top <= inner.top and outer.right >= inner.right and outer.bottom >= inner.bottom


def intersects(a: Rect, b: Rect) -> bool:
    return a.left < b.right and b.left < a.right and a.top < b.bottom and b.top < a.bottom


def distance_to_rect(x: int, y: int, r: Rect) -> float:
    """0 inside, else Euclidean distance to the nearest edge."""
    dx = max(r.left - x, 0, x - r.right)
    dy = max(r.top - y, 0, y - r.bottom)
    return math.hypot(dx, dy)


def main_child(target: UIElement, others: list[UIElement]) -> UIElement | None:
    """Largest strictly-contained child covering at least MAIN_CHILD_MIN_SHARE of the target."""
    area = target.rect.area
    if area <= 0:
        return None
    kids = [
        o for o in others
        if o is not target and o.hwnd == target.hwnd and o.rect.area < area and contains(target.rect, o.rect)
        and o.rect.area >= MAIN_CHILD_MIN_SHARE * area
    ]
    return max(kids, key=lambda o: o.rect.area) if kids else None


def obstacles(target: UIElement, others: list[UIElement]) -> list[Rect]:
    """Boxes to keep away from: overlapping elements that neither contain the target
    (ancestors) nor are its main part. Small children (a tab's close button) count."""
    out = []
    for o in others:
        if o is target or o.is_window or o.rect.area <= 0:
            continue
        if not intersects(o.rect, target.rect):
            continue
        if contains(o.rect, target.rect) and o.rect.area >= target.rect.area:
            continue  # ancestor
        out.append(o.rect)
    return out


def click_point(target: UIElement, others: list[UIElement], depth: int = 0) -> tuple[int, int]:
    """Return (x, y). Descends into the main child first (max 3 levels), then maximises the
    clearance from obstacles inside an inset of the target."""
    if depth < 3:
        child = main_child(target, others)
        if child is not None:
            return click_point(child, others, depth + 1)
    r = target.rect
    obs = obstacles(target, others)
    cx, cy = r.center
    if not obs or r.width < 3 * EDGE_MARGIN or r.height < 3 * EDGE_MARGIN:
        return cx, cy
    mx = min(EDGE_MARGIN, r.width // 4)
    my = min(EDGE_MARGIN, r.height // 4)
    best = (cx, cy)
    best_key = (-1.0, 0.0)
    for i in range(GRID):
        x = r.left + mx + (r.width - 2 * mx) * i // (GRID - 1)
        for j in range(GRID):
            y = r.top + my + (r.height - 2 * my) * j // (GRID - 1)
            clearance = min(distance_to_rect(x, y, o) for o in obs)
            key = (round(clearance, 1), -math.hypot(x - cx, y - cy))
            if key > best_key:
                best_key, best = key, (x, y)
    return best
