from laya_agent.actuation.targeting import click_point, main_child, obstacles
from laya_agent.models import Rect, UIElement


def _el(i, name, l, t, r, b, kind="Button", hwnd=1):
    return UIElement(id=i, kind=kind, name=name, rect=Rect(l, t, r, b), hwnd=hwnd)


# Windows Terminal geometry, roughly: a 24px-tall tab strip.
TAB = _el(1, "laya-agent", 200, 100, 440, 130, kind="TabItem")
CLOSE = _el(2, "Close Tab", 410, 106, 428, 124)  # 18px button at the tab's right end
SPLIT = _el(3, "New Tab", 820, 100, 910, 130, kind="SplitButton")
PRIMARY = _el(4, "PrimaryButton", 820, 100, 868, 130)
DROPDOWN = _el(5, "", 880, 100, 910, 130)
WINDOW = _el(6, "Terminal", 0, 0, 2000, 1000, kind="Window")
ALL = [TAB, CLOSE, SPLIT, PRIMARY, DROPDOWN, WINDOW]


def test_split_button_clicks_its_primary_part_not_the_divider():
    assert main_child(SPLIT, ALL) is PRIMARY
    x, y = click_point(SPLIT, ALL)
    assert PRIMARY.rect.left < x < PRIMARY.rect.right and abs(y - 115) <= 2
    assert x < DROPDOWN.rect.left - 10  # clear of the dropdown arrow


def test_tab_click_keeps_clear_of_its_close_button():
    assert obstacles(TAB, ALL) == [CLOSE.rect]
    x, y = click_point(TAB, ALL)
    assert TAB.rect.left <= x <= TAB.rect.right and TAB.rect.top <= y <= TAB.rect.bottom
    assert x < CLOSE.rect.left - 30  # well away from the close button
    assert abs(y - 115) <= 4  # still vertically centred


def test_close_button_itself_is_hit_at_centre():
    assert click_point(CLOSE, ALL) == CLOSE.rect.center


def test_window_and_ancestors_are_not_obstacles():
    assert obstacles(CLOSE, ALL) == []  # tab contains it, window is a window


def test_lonely_element_uses_centre():
    lone = _el(9, "OK", 10, 10, 110, 50)
    assert click_point(lone, [lone]) == (60, 30)
