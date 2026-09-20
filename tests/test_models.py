from laya_agent.models import Rect, UIElement, Snapshot, Decision, META_ACTIONS


def _el(i=1, name="OK", kind="Button", path=""):
    return UIElement(id=i, kind=kind, name=name, rect=Rect(10, 20, 110, 60), path=path)


def test_rect_center_and_area():
    r = Rect(10, 20, 110, 60)
    assert r.center == (60, 40)
    assert r.area == 4000


def test_element_label_with_path():
    assert _el().label() == "Button 'OK'"
    assert _el(path="Dialog > Footer").label() == "Button 'OK' in Dialog > Footer"


def test_snapshot_by_id():
    snap = Snapshot(png=b"", width=1, height=1, window_title="w", elements=[_el(1), _el(2, "Cancel")])
    assert snap.by_id(2).name == "Cancel"
    assert snap.by_id(9) is None


def test_decision_is_click():
    d = Decision(action="click_1", element=_el(), confidence=0.9, done_prob=0.1, top_k=[])
    assert d.is_click
    d2 = Decision(action="done", element=None, confidence=0.9, done_prob=0.9, top_k=[])
    assert not d2.is_click and "done" in META_ACTIONS
