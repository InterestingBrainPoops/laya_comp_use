from laya_agent.brain.aliases import AliasStore
from laya_agent.brain.lexical import lexical_scores, score_element, token_similarity, tokens
from laya_agent.models import Rect, UIElement


def _el(i, name, kind="TabItem"):
    return UIElement(id=i, kind=kind, name=name, rect=Rect(0, 0, 1, 1))


TABS = [_el(1, "New Tab", "Button"), _el(2, "NandhaKishorM/laya"), _el(3, "convaiinnovations/laya · Hugging Face"),
        _el(4, "Install GitHub", "Button")]


def test_tokens_drop_ui_stop_words():
    assert tokens("switch to the NandhaKishorM tab") == ["nandhakishorm"]


def test_token_similarity_levels():
    assert token_similarity("laya", "laya") == 1.0
    assert token_similarity("nandha", "nandhakishorm") == 0.85
    assert token_similarity("github", "hugging") == 0.0


def test_explicit_name_wins_over_new_tab():
    s = lexical_scores("switch to the NandhaKishorM tab", TABS)
    assert s[2] >= 0.9 and s[1] == 0.0 and s[3] < 0.5


def test_shared_token_is_ambiguous():
    s = lexical_scores("open the laya tab", TABS)
    assert s[2] > 0.5 and s[3] > 0.5 and abs(s[2] - s[3]) < 0.25


def test_no_match_scores_zero_and_kind_mismatch_is_penalised():
    s = lexical_scores("switch to the github tab", TABS)
    assert s[2] == 0.0  # the tab does not say github
    assert 0.0 < s[4] < 0.9  # 'Install GitHub' matches the word but is a button, not a tab
    assert lexical_scores("click install github", TABS)[4] >= 0.9


def test_alias_makes_vague_goal_explicit(tmp_path):
    store = AliasStore(tmp_path / "a.json")
    store.remember("switch to the github tab", "NandhaKishorM/laya")
    s = lexical_scores("go to the github tab", TABS, store.as_dict())
    assert s[2] >= 0.9
    reloaded = AliasStore(tmp_path / "a.json")
    assert reloaded.as_dict() == {"NandhaKishorM/laya": ["github"]}


def test_score_element_distinctive_token():
    assert score_element(["nandhakishorm"], _el(1, "NandhaKishorM/laya")) >= 0.9
    assert score_element(["new"], _el(1, "New Tab")) >= 0.9  # full coverage, exact token
    assert score_element(["new", "window"], _el(1, "New Tab")) < 0.9  # half the goal unmatched


def test_kind_word_inside_name_is_not_penalised():
    s = lexical_scores("open a new tab", TABS)
    assert s[1] >= 0.9  # button 'New Tab' although the goal says "tab"
