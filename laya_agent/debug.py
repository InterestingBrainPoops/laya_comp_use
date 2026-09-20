"""Debug rendering: annotate a screenshot with every parsed element. Framework-free (PIL)."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from laya_agent.models import Snapshot

COLOR_ALL = (140, 140, 140)
COLOR_SHORTLIST = (255, 149, 0)
COLOR_FINE = (52, 199, 89)
COLOR_CHOSEN = (255, 59, 48)


def _font(size: int):
    try:
        return ImageFont.truetype("segoeui.ttf", size)
    except Exception:
        return ImageFont.load_default()


def annotate_png(
    snap: Snapshot,
    shortlist: set[int] | None = None,
    fine: set[int] | None = None,
    chosen: int | None = None,
) -> bytes:
    """Every element gets a grey box and its id. Shortlisted orange, fine-pass green, chosen red."""
    shortlist = shortlist or set()
    fine = fine or set()
    img = Image.open(io.BytesIO(snap.png)).convert("RGB") if snap.png else Image.new("RGB", (snap.width, snap.height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(18)
    for e in snap.elements:
        if e.id == chosen:
            color, width = COLOR_CHOSEN, 4
        elif e.id in fine:
            color, width = COLOR_FINE, 3
        elif e.id in shortlist:
            color, width = COLOR_SHORTLIST, 3
        else:
            color, width = COLOR_ALL, 1
        r = e.rect
        draw.rectangle([r.left, r.top, r.right, r.bottom], outline=color, width=width)
        label = str(e.id)
        tw = draw.textlength(label, font=font)
        y = max(0, r.top - 20)
        draw.rectangle([r.left, y, r.left + tw + 6, y + 20], fill=color)
        draw.text((r.left + 3, y), label, fill="white", font=font)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def element_listing(snap: Snapshot, shortlist: set[int] | None = None, fine: set[int] | None = None) -> str:
    shortlist = shortlist or set()
    fine = fine or set()
    lines = [f"window {snap.window_title!r}: {len(snap.elements)} elements ({snap.source})"]
    for e in snap.elements:
        mark = "**" if e.id in fine else "*" if e.id in shortlist else " "
        lines.append(f"{mark:>2} [{e.id:>3}] {e.label()} @{e.center}")
    return "\n".join(lines)
