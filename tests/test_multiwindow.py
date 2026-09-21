"""Cross-window behaviour: window elements, background elements, lexical window matching."""
from laya_agent.actuation.executor import Executor
from laya_agent.brain.lexical import lexical_scores
from laya_agent.brain.laya_policy import LayaPolicy
from laya_agent.config import Config
from laya_agent.models import Decision, Rect, Snapshot, UIElement
from tests.test_policy import _fake_predict

R = Rect(0, 0, 10, 10)
CHROME_WIN = UIElement(id=1, kind="Window", name="GitHub - Google Chrome", rect=R, hwnd=11, window="Google Chrome", foreground=True, selected=True)
NEW_TAB = UIElement(id=2, kind="Button", name="New Tab", rect=R, hwnd=11, window="Google Chrome", foreground=True)
SPOTIFY_WIN = UIElement(id=3, kind="Window", name="Spotify Premium", rect=R, hwnd=22, window="Spotify Premium", foreground=False)
PLAY = UIElement(id=4, kind="Button", name="Play", rect=R, hwnd=22, window="Spotify Premium", foreground=False)
DISCORD_MIN = UIElement(id=5, kind="Window", name="Discord", rect=Rect(0, 0, 0, 0), hwnd=33, window="Discord", foreground=False, minimized=True)
SONG_WIN = UIElement(id=6, kind="Window", name="grandson - Overdose", rect=R, hwnd=44, window="Spotify", foreground=False)
ALL = [CHROME_WIN, NEW_TAB, SPOTIFY_WIN, PLAY, DISCORD_MIN]


def test_labels_name_the_window_for_background_elements():
    assert NEW_TAB.label() == "button 'New Tab'"
    assert PLAY.label() == "button 'Play' in window Spotify Premium"
    assert SPOTIFY_WIN.label() == "window 'Spotify Premium'"
    assert DISCORD_MIN.label() == "window 'Discord' (minimized)"
    assert SONG_WIN.label() == "window 'grandson - Overdose' of app Spotify"


def test_app_name_resolves_window_whose_title_is_a_song():
    s = lexical_scores("switch to spotify", [CHROME_WIN, NEW_TAB, SONG_WIN, PLAY])
    assert s[6] >= 0.9 and s[6] - max(s[1], s[2], s[4]) >= 0.25


NEWTAB_WIN = UIElement(id=7, kind="Window", name="New Tab - Google Chrome", rect=R, hwnd=11, window="Chrome", foreground=True)


def test_window_matched_only_by_title_loses_to_the_element():
    s = lexical_scores("open a new tab in chrome", [CHROME_WIN, NEW_TAB, SPOTIFY_WIN, PLAY, NEWTAB_WIN])
    assert s[2] >= 0.9 and s[2] - s[1] >= 0.25  # the button, not the window titled 'GitHub - Google Chrome'
    assert s[2] - s[7] >= 0.25  # nor the window titled 'New Tab - Google Chrome'
    s2 = lexical_scores("switch to chrome", [CHROME_WIN, NEW_TAB])
    assert s2[1] >= 0.9 and s2[1] - s2[2] >= 0.25  # the window, not the button


def test_goal_naming_only_the_app_prefers_the_window():
    s = lexical_scores("switch to spotify", ALL)
    assert s[3] >= 0.9 and s[3] - s[4] >= 0.25


def test_goal_naming_element_and_app_prefers_the_element():
    s = lexical_scores("press play in spotify", ALL)
    assert s[4] > s[3] and s[4] >= 0.9


def test_policy_switches_window_by_name():
    snap = Snapshot(png=b"", width=1, height=1, window_title="GitHub - Google Chrome", elements=ALL, windows=["GitHub - Google Chrome", "Spotify Premium", "Discord"])
    d = LayaPolicy(Config(alias_path=None), predict=_fake_predict("click_2")).decide("open discord", snap, history=[])
    assert d.element is DISCORD_MIN and d.raw["_decided_by"] == "name match"


def test_executor_activates_windows_and_background_clicks(monkeypatch):
    ex = Executor(Config(dry_run=False))
    calls = []
    monkeypatch.setattr(ex, "activate", lambda hwnd: calls.append(("activate", hwnd)))
    monkeypatch.setattr(ex, "click_at", lambda x, y, double=False: calls.append(("click", (x, y))))
    note = ex.execute(Decision(action="click_3", element=SPOTIFY_WIN, confidence=1, done_prob=0, top_k=[]))
    assert note.startswith("switched to window") and calls == [("activate", 22)]
    calls.clear()
    ex.execute(Decision(action="click_4", element=PLAY, confidence=1, done_prob=0, top_k=[]))
    assert calls == [("activate", 22), ("click", PLAY.center)]
    calls.clear()
    ex.execute(Decision(action="click_2", element=NEW_TAB, confidence=1, done_prob=0, top_k=[]))
    assert calls == [("click", NEW_TAB.center)]  # foreground: no activation needed
