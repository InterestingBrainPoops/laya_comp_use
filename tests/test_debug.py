import io

from PIL import Image

from laya_agent.debug import annotate_png, element_listing
from laya_agent.models import Rect, Snapshot, UIElement


def _snap():
    els = [UIElement(id=i, kind="Button", name=f"B{i}", rect=Rect(10 * i, 10, 10 * i + 8, 30)) for i in range(1, 4)]
    return Snapshot(png=b"", width=60, height=40, window_title="w", elements=els)


def test_annotate_returns_png_of_screen_size():
    png = annotate_png(_snap(), shortlist={1, 2}, fine={2}, chosen=2)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG" and img.size == (60, 40)


def test_listing_marks_shortlist_and_fine():
    text = element_listing(_snap(), shortlist={1, 2}, fine={2})
    lines = text.splitlines()
    assert lines[0].startswith("window 'w': 3 elements")
    assert lines[1].startswith(" *") and lines[2].startswith("**") and lines[3].startswith("  ")
